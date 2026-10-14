/* V1-3: 保留 V1-1 全部功能（$G2 控制台/追踪协议），
 *       新增 PS2 无线手柄——小车总线电机控制（只做小车，云台暂不接）。
 *
 * 硬件: KPZ-ESP32 (DOIT ESP32 DEVKIT V1), Espressif core 3.3.12
 *   UART2 RX=16 TX=17 115200 8N1  -> 板载总线接口(合成 DAT) + OpenMV（$G2 协议，原有）
 *   小车总线(ZMotorD 驱动): 插板载"总线接口"(GND/VIN/DAT)，与 OpenMV 共用 UART2
 *   PS2 接收器: DAT=19 CMD=18 ATT=15 CLK=23（可改）
 *
 * 手柄控制（移植自 Carbot 出厂代码 z_main.py / car_run.py）:
 *   车模式: 左摇杆前推/后拉 = 左电机; 右摇杆前推/后拉 = 右电机（出厂红灯模式公式）
 *           左摇杆按下(L3) = 原地旋转; 右摇杆按下(R3) = 原地旋转(反向)
 *           START = 停车
 *   云台模式: SELECT 切换 车<->云台
 *           左摇杆 Y = tilt(俯仰/轴2), 右摇杆 X = pan(水平/轴1) 手动控制($G2 JOG)
 *           START = 回中($G2 HOME)
 *   进云台模式 = 发 $G2 HOLD (停止自动识别, 手动接管)
 *   回车模式   = 发 $G2 TRACK (恢复自动识别靶心)
 *   小车协议: #006P1500T1000!...  P=1500±速度, T=运行时间ms(0=持续)
 *
 * 说明:
 *   - SELECT 原为出厂智能模式切换(循迹/避障), 本板未接传感器, 现用作 车<->云台 切换。
 *   - 与原厂一致: 仅红 ////// //     ， 灯模式(红绿都亮)下摇杆有效, 按手柄 MODE 键切换。
 */
#include <Arduino.h>
#include <string.h>

// ===================== 配置区（按实际接线修改） =====================
// PS2 手柄接收器引脚（DAT/CMD/ATT/CLK 对应接收器上 DI/DO/CS/CK）
constexpr uint8_t PS2_DAT = 19, PS2_CMD = 18, PS2_ATT = 15, PS2_CLK = 23;
// ---- 小车总线串口（下位机）----
// KPZ-ESP32 板: 电机驱动插板载"总线接口"(GND/VIN/DAT)，板上串口 RX/TX 经电路合成 DAT。
// 总线与 OpenMV 共用 UART2(RX16/TX17 115200)，电机指令 #...! 与 $G2 各认各的，不冲突。
// 小车总线通道: 你的实际接线(两个 ZMotorD 双路驱动)
// 双路驱动占2个ID: 第1个ID=左电机, 第2个ID=右电机
//   驱动A(ID6): 006=左下轮, 007=右下轮
//   驱动B(ID9): 009=左上轮, 010=右上轮
// 左摇杆同时驱动 006+009(左侧), 右摇杆同时驱动 007+010(右侧)
constexpr uint8_t CAR_CH_LB = 6, CAR_CH_RB = 7, CAR_CH_LT = 9, CAR_CH_RT = 10;
constexpr bool   CAR_BUS_BRACES = false;       // ZMotorD 手册格式是 #idPpwmTtime!, 不带{}
// 每个轮子独立方向: 推摇杆"上"应=车前进。若某轮转向相反(车斜着转圈), 把对应 true<->false 翻转
constexpr bool   CAR_INV_LB = false, CAR_INV_RB = true, CAR_INV_LT = false, CAR_INV_RT = true;
constexpr float  CAR_STICK_SCALE = 7.8f;       // 摇杆偏移 -> P 偏移(出厂值)
constexpr int16_t CAR_P_DEADZONE = 150;        // P 值死区, 小于该偏差视为停止(出厂值)
constexpr int16_t CAR_SPIN_SPEED = 700;        // L3/R3 原地旋转速度(出厂 car_speed=700)
constexpr uint32_t PS2_POLL_MS = 20;           // 手柄轮询间隔
// ---- 云台手柄控制（SELECT 切换 车模式<->云台模式）----
// 云台模式: 右摇杆 X=pan(水平/轴1), Y=tilt(俯仰/轴2); START=回中(HOME)
// 进云台模式发 $G2 HOLD(停止自动识别), 切回车模式发 $G2 TRACK(恢复自动识别)
constexpr uint32_t GIMBAL_POLL_MS = 100;       // 手动 JOG 发送间隔(ms)
constexpr int16_t  GIMBAL_DEADZONE = 20;       // 摇杆死区(0~255, 中=128)
constexpr int16_t  GIMBAL_MAX_DELTA = 60;      // 单次 JOG 最大步进 us
// ================================================================

// PS2 按键位（bit=0 表示按下, active-low）
// 数据帧: buf[3]=按键字节1, buf[4]=按键字节2; buttons()=(buf[3]<<8)|buf[4]
// 标准 PS2 协议: SELECT/L3/R3/START/方向键 在 buf[3](bit8..15), L2/R2/L1/R1/△○×□ 在 buf[4](bit0..7)
enum : uint16_t {
  PS2_L2 = 1 << 0, PS2_R2 = 1 << 1, PS2_L1 = 1 << 2, PS2_R1 = 1 << 3,
  PS2_TRI = 1 << 4, PS2_CIRC = 1 << 5, PS2_CROSS = 1 << 6, PS2_SQUARE = 1 << 7,
  PS2_SELECT = 1 << 8, PS2_L3 = 1 << 9, PS2_R3 = 1 << 10, PS2_START = 1 << 11,
  PS2_UP = 1 << 12, PS2_RIGHT = 1 << 13, PS2_DOWN = 1 << 14, PS2_LEFT = 1 << 15
};

// ---------- PS2 位操作驱动（移植自 Carbot C3 出厂代码 z_ps2.py） ----------
class PS2Controller {
 public:
  PS2Controller(uint8_t dat, uint8_t cmd, uint8_t att, uint8_t clk)
      : dat_(dat), cmd_(cmd), att_(att), clk_(clk) {
    memset(buf_, 0, sizeof(buf_));
  }

  // 必须在 setup() 里调用（避免全局构造阶段操作 GPIO）
  void begin() {
    pinMode(dat_, INPUT_PULLUP);  // 接收器 DO 线上拉, 防止悬空误读
    pinMode(cmd_, OUTPUT);
    pinMode(att_, OUTPUT);
    pinMode(clk_, OUTPUT);
    digitalWrite(cmd_, HIGH);
    digitalWrite(clk_, HIGH);
    digitalWrite(att_, HIGH);
  }

  // 拉低 ATT 后发 0x01/0x42 命令, 读回 9 字节状态
  void writeRead() {
    digitalWrite(att_, LOW);
    buf_[0] = transfer(0x01);
    buf_[1] = transfer(0x42);
    for (int i = 2; i < 9; ++i) buf_[i] = transfer(buf_[0]);
    digitalWrite(att_, HIGH);
  }

  // 原厂 data_is_ready(): 仅红灯模式(0x73)帧有效(0xff+0x73+0x5a)。
  // 严格要求模式字节: 未配对/乱码帧(buf1≠0x73)不触发, 否则手柄没开会乱动电机。
  bool dataReady() const {
    return buf_[0] == 0xFF && buf_[1] == 0x73 && buf_[2] == 0x5A;
  }
  uint8_t modeByte() const { return buf_[1]; }
  uint8_t byte(size_t i) const { return (i < sizeof(buf_)) ? buf_[i] : 0; }
  uint16_t buttons() const { return (uint16_t(buf_[3]) << 8) | buf_[4]; }  // bit=0 按下
  bool pressed(uint16_t mask) const { return (buttons() & mask) == 0; }
  int16_t lx() const { return buf_[7]; }  // 左摇杆 X 0~255(中=128)
  int16_t ly() const { return buf_[8]; }  // 左摇杆 Y 0(上)~255(下)
  int16_t rx() const { return buf_[5]; }
  int16_t ry() const { return buf_[6]; }

 private:
  uint8_t transfer(uint8_t data) {  // LSB 先发, 8bit 位同步
    uint8_t readData = 0;
    for (uint8_t i = 0; i < 8; ++i) {
      digitalWrite(cmd_, (data & (0x01 << i)) ? HIGH : LOW);
      digitalWrite(clk_, HIGH);
      delayMicroseconds(5);
      digitalWrite(clk_, LOW);
      delayMicroseconds(5);
      if (digitalRead(dat_) == HIGH) readData |= 0x01 << i;
    }
    digitalWrite(clk_, HIGH);
    delayMicroseconds(10);
    return readData;
  }
  uint8_t dat_, cmd_, att_, clk_;
  uint8_t buf_[9];
};

// ---------- 小车总线函数前置声明（供串口测试命令调用） ----------
void sendCarP(int16_t lP, int16_t rP, uint16_t runtime);
void carStop();

// ---------- 原有 $G2 协议与控制台（以下保持原逻辑，未改动） ----------
constexpr int BUS_RX = 16, BUS_TX = 17;
constexpr uint32_t BAUD = 115200, TIMEOUT_MS = 1000, POLL_MS = 2000;
HardwareSerial Bus(2);
char rx[192] = {}, input[40] = {}, expected[48] = {}, errorPrefix[32] = {};
size_t rxUsed = 0, inputUsed = 0;
bool dropping = false, pending = false, polling = true, nextInfo = true;
bool pendingPoll = false;
uint16_t sequence = 0;
uint32_t sentAt = 0, lastRx = 0, lastPoll = 0;
uint32_t requests = 0, replies = 0, timeouts = 0, writeErrors = 0;
uint32_t laserHitCount = 0;

void help() {
  Serial.println("V1-1 + V1-3 console; newline commands:");
  Serial.println("start | track | stop/hold | off | home | calpan");
  Serial.println("ping | status | info | limits | stats | pollon | polloff");
  Serial.println("pan+ | pan- | tilt+ | tilt- (HOLD only, 20us)");
  Serial.println("probe pan+/- | probe tilt+/- | accept");
  Serial.println("PS2: SELECT=车/云台切换, L-stick/R-stick=左右电机, L3/R3=旋转, START=停车");
  Serial.println("GIMBAL: L-stick Y=tilt(俯仰), R-stick X=pan(水平), START=回中; 进云台=HOLD 回车=TRACK");
  Serial.println("CarBus test: cf=forward 1s, cb=backward 1s, cs=stop, ch <id> <pwm> <time>=single channel");
}

void sendCommand(const char *verb, const char *args = nullptr, bool poll = false) {
  const bool priority = !strcmp(verb, "HOLD") || !strcmp(verb, "OFF");
  if (pending && !priority && !(pendingPoll && !poll)) {
    Serial.println("BUSY: wait for reply/timeout."); return;
  }
  ++sequence;
  char message[64];
  if (args) snprintf(message, sizeof(message), "$G2,%s,%u,%s!", verb, unsigned(sequence), args);
  else snprintf(message, sizeof(message), "$G2,%s,%u!", verb, unsigned(sequence));
  if (!strcmp(verb, "PING")) snprintf(expected, sizeof(expected), "$G2,PONG,%u!", unsigned(sequence));
  else snprintf(expected, sizeof(expected), "$G2,ACK,%u,%s,", unsigned(sequence),
                !strcmp(verb, "INFO") ? "INFO" : !strcmp(verb, "LIMITS") ? "LIMITS" : "STATE");
  snprintf(errorPrefix, sizeof(errorPrefix), "$G2,ERR,%u,", unsigned(sequence));
  rxUsed = 0;
  sentAt = lastPoll = millis();
  pending = true; pendingPoll = poll; ++requests;
  const size_t length = strlen(message);
  if (Bus.write(reinterpret_cast<const uint8_t *>(message), length) != length) {
    pending = false; ++writeErrors;
    Serial.println("UART_WRITE_FAILED; execution state unknown."); return;
  }
  Serial.print("TX "); Serial.println(message);
}

bool parseInfo(const char *s, uint32_t *v) {
  for (size_t i = 0; i < 7; ++i) {
    if (*s < '0' || *s > '9') return false;
    uint32_t n = 0;
    while (*s >= '0' && *s <= '9') {
      uint32_t digit = uint32_t(*s++ - '0');
      if (n > (UINT32_MAX - digit) / 10) return false;
      n = n * 10 + digit;
    }
    v[i] = n;
    if (*s++ != (i == 6 ? '!' : ',')) return false;
  }
  return !*s && v[4] <= 1;
}

void completeFrame() {
  if (!pending) return;
  const bool error = !strncmp(rx, errorPrefix, strlen(errorPrefix));
  const bool pong = expected[strlen(expected) - 1] == '!';
  const bool match = pong ? !strcmp(rx, expected) : !strncmp(rx, expected, strlen(expected));
  if (!error && !match) return; 
  uint32_t v[7];
  const bool info = !error && strstr(expected, ",INFO,");
  if (info && !parseInfo(rx + strlen(expected), v)) return;
  pending = false; ++replies;
  Serial.print(error ? "REJECT " : "REPLY "); Serial.println(rx);
  if (info) {
    Serial.print("INFO camera="); Serial.print(v[4] ? "OK" : "ERROR");
    Serial.print(" uptime_ms="); Serial.print(v[0]);
    Serial.print(" frames="); Serial.print(v[1]);
    Serial.print(" fps_x10="); Serial.print(v[2]);
    Serial.print(" snap_us="); Serial.print(v[3]);
    Serial.print(" cam_err="); Serial.print(v[5]);
    Serial.print(" uart_err="); Serial.print(v[6]);
    Serial.print(" rtt_ms="); Serial.println(millis() - sentAt);
  }
}

void receiveByte(char value) {
  lastRx = millis();
  if (value == '$') { rxUsed = 0; rx[rxUsed++] = value; }
  else if (rxUsed) {
    if (rxUsed >= sizeof(rx) - 1) { rxUsed = 0; return; }
    rx[rxUsed++] = value;
    if (value == '!') { 
      rx[rxUsed] = '\0'; 
      if (strncmp(rx, "$G2,HIT,1!", 10) == 0) {
        laserHitCount++;
        Serial.print("\n>>> LASER HIT TARGET! Total Hits: ");
        Serial.println(laserHitCount);
        rxUsed = 0;
        return; 
      }
      completeFrame(); 
      rxUsed = 0; 
    }
  }
}

void command(const char *line) {
  if (!strcmp(line, "start")) sendCommand("START");
  else if (!strcmp(line, "track")) sendCommand("TRACK");
  else if (!strcmp(line, "stop") || !strcmp(line, "hold")) sendCommand("HOLD");
  else if (!strcmp(line, "off")) sendCommand("OFF");
  else if (!strcmp(line, "home")) sendCommand("HOME");
  else if (!strcmp(line, "calpan")) sendCommand("CALPAN");
  else if (!strcmp(line, "ping") || !strcmp(line, "p")) sendCommand("PING");
  else if (!strcmp(line, "status")) sendCommand("STATUS");
  else if (!strcmp(line, "info")) sendCommand("INFO");
  else if (!strcmp(line, "limits")) sendCommand("LIMITS");
  else if (!strcmp(line, "pan+")) sendCommand("JOG", "1,20");
  else if (!strcmp(line, "pan-")) sendCommand("JOG", "1,-20");
  else if (!strcmp(line, "tilt+")) sendCommand("JOG", "2,20");
  else if (!strcmp(line, "tilt-")) sendCommand("JOG", "2,-20");
  else if (!strcmp(line, "accept")) sendCommand("ACCEPT");
  else if (!strcmp(line, "cf")) sendCarP(2000, 2000, 1000);  // 小车总线测试: 前进1秒(自然方向)
  else if (!strcmp(line, "cb")) sendCarP(1000, 1000, 1000);  // 小车总线测试: 后退1秒
  else if (!strcmp(line, "cs")) carStop();                   // 小车总线测试: 停车
  else if (!strncmp(line, "ch ", 3)) {                       // 单通道测试: ch <id> <pwm> <time>
    int id = 0, pwm = 0, t = 0;
    if (sscanf(line + 3, "%d %d %d", &id, &pwm, &t) == 3 &&
        id >= 0 && id <= 254 && pwm >= 500 && pwm <= 2500 && t >= 0 && t <= 9999) {
      char b[24];
      snprintf(b, sizeof(b), "#%03dP%04dT%04d!", id, pwm, t);
      Serial.print("BUS> "); Serial.println(b);
      Bus.write(reinterpret_cast<const uint8_t *>(b), strlen(b));
    } else Serial.println("usage: ch <id000-254> <pwm500-2500> <time0000-9999>");
  }
  else if (!strcmp(line, "pollon")) { polling = true; lastPoll = millis(); Serial.println("POLL ON"); }
  else if (!strcmp(line, "polloff")) { polling = false; Serial.println("POLL OFF"); }
  else if (!strcmp(line, "stats")) {
    Serial.print("STATS req="); Serial.print(requests);
    Serial.print(" rep="); Serial.print(replies);
    Serial.print(" timeouts="); Serial.print(timeouts);
    Serial.print(" laser_hits="); Serial.println(laserHitCount);
  } else if (*line) help();
}

// ===================== 新增: 手柄与小车控制 =====================
PS2Controller ps2(PS2_DAT, PS2_CMD, PS2_ATT, PS2_CLK);
uint32_t lastPs2Poll = 0, lastValidPs2 = 0;
bool startWasPressed = true;  // 防止上电时按住 START 误触发

// ---- 车/云台模式切换 ----
enum : uint8_t { MODE_CAR = 0, MODE_GIMBAL = 1 };
uint8_t gimbalMode = MODE_CAR;
const char *pendingGimbalCmd = nullptr;  // 待发的 HOLD/TRACK(模式切换), 总线空闲后发出
bool selWasPressed = true;               // 防止上电时按住 SELECT 误触发
uint32_t lastGimbalSend = 0;             // 云台 JOG 节流

// ===================== 新增: 小车总线驱动（移植自 car_run.py） =====================
// 说明: 电机指令经板载总线接口发出, 与 $G2 共用 Bus(UART2 16/17)。
int16_t lastCarPL = 1500, lastCarPR = 1500;

// 发送小车总线指令(4通道): lP/rP 为"自然前进"方向偏移(1500+速度), 各通道按物理方向独立取反
// 左路=LB(006)+LT(009), 右路=RB(007)+RT(010); CAR_BUS_BRACES 决定是否带 {...}
void sendCarP(int16_t lP, int16_t rP, uint16_t runtime) {
  int16_t pLB = CAR_INV_LB ? 1500 - (lP - 1500) : lP;
  int16_t pRB = CAR_INV_RB ? 1500 - (rP - 1500) : rP;
  int16_t pLT = CAR_INV_LT ? 1500 - (lP - 1500) : lP;
  int16_t pRT = CAR_INV_RT ? 1500 - (rP - 1500) : rP;
  char buf[96];
  // 注意: runtime 在格式串里出现4次, 必须传4次, 否则字段错位
  if (CAR_BUS_BRACES) {
    snprintf(buf, sizeof(buf), "{#%03dP%04dT%04d!#%03dP%04dT%04d!#%03dP%04dT%04d!#%03dP%04dT%04d!}",
             CAR_CH_LB, pLB, runtime, CAR_CH_RB, pRB, runtime,
             CAR_CH_LT, pLT, runtime, CAR_CH_RT, pRT, runtime);
  } else {
    snprintf(buf, sizeof(buf), "#%03dP%04dT%04d!#%03dP%04dT%04d!#%03dP%04dT%04d!#%03dP%04dT%04d!",
             CAR_CH_LB, pLB, runtime, CAR_CH_RB, pRB, runtime,
             CAR_CH_LT, pLT, runtime, CAR_CH_RT, pRT, runtime);
  }
  Serial.print("CAR> "); Serial.println(buf);   // 调试: 打印发往小车总线的指令
  Bus.write(reinterpret_cast<const uint8_t *>(buf), strlen(buf));  // 经板载总线接口(UART2)发出
}

// 摇杆值(0~255, 中=128) -> P 值(出厂公式: 1500 + 7.8*(128-Y))
int16_t stickToP(int16_t stickY) {
  return 1500 + int16_t(CAR_STICK_SCALE * (128 - stickY));
}

// 停车(出厂 car_stop 风格: T1000 缓停)
void carStop() { sendCarP(1500, 1500, 1000); }

void handleCar() {
  // L3/R3 原地旋转优先于摇杆
  if (ps2.pressed(PS2_L3)) {
    if (lastCarPL != 1500 - CAR_SPIN_SPEED || lastCarPR != 1500 - CAR_SPIN_SPEED) {
      lastCarPL = lastCarPR = 1500 - CAR_SPIN_SPEED;
      sendCarP(lastCarPL, lastCarPR, 0);
    }
    return;
  }
  if (ps2.pressed(PS2_R3)) {
    if (lastCarPL != 1500 + CAR_SPIN_SPEED || lastCarPR != 1500 + CAR_SPIN_SPEED) {
      lastCarPL = lastCarPR = 1500 + CAR_SPIN_SPEED;
      sendCarP(lastCarPL, lastCarPR, 0);
    }
    return;
  }

  int16_t lP = stickToP(ps2.ly());  // 自然前进方向偏移, 各通道取反在 sendCarP 里处理
  int16_t rP = stickToP(ps2.ry());
  if (abs(lP - 1500) < CAR_P_DEADZONE) lP = 1500;
  if (abs(rP - 1500) < CAR_P_DEADZONE) rP = 1500;
  if (lP != lastCarPL || rP != lastCarPR) {  // 值变化才发, 避免刷总线
    lastCarPL = lP; lastCarPR = rP;
    sendCarP(lP, rP, 0);
  }
}

// 云台手动控制: 左摇杆 Y=tilt(俯仰/轴2, 上推=+), 右摇杆 X=pan(水平/轴1, 右推=+)
// 每 GIMBAL_POLL_MS 发一次 $G2 JOG; $G2 总线一次只能挂一个未回命令, 双轴同时推时交替发送(各约 200ms 一次)
void handleGimbal() {
  if (pending || millis() - lastGimbalSend < GIMBAL_POLL_MS) return;
  lastGimbalSend = millis();
  int16_t panD = ps2.rx() - 128;    // 右摇杆 X -> pan(轴1), 右推=+
  int16_t tiltD = 128 - ps2.ly();   // 左摇杆 Y -> tilt(轴2), 上推=+
  if (abs(panD) < GIMBAL_DEADZONE) panD = 0;
  if (abs(tiltD) < GIMBAL_DEADZONE) tiltD = 0;
  panD = constrain(panD / 2, -int(GIMBAL_MAX_DELTA), int(GIMBAL_MAX_DELTA));
  tiltD = constrain(tiltD / 2, -int(GIMBAL_MAX_DELTA), int(GIMBAL_MAX_DELTA));
  static uint8_t gimbalTurn = 0;    // 0=pan 优先, 1=tilt 优先(交替)
  char arg[12];
  if (panD && (!tiltD || gimbalTurn == 0)) {
    snprintf(arg, sizeof(arg), "1,%d", panD);
    sendCommand("JOG", arg);
  } else if (tiltD) {
    snprintf(arg, sizeof(arg), "2,%d", tiltD);
    sendCommand("JOG", arg);
  }
  gimbalTurn = 1 - gimbalTurn;
}

void setup() {
  Serial.begin(BAUD);
  Bus.begin(BAUD, SERIAL_8N1, BUS_RX, BUS_TX);  // 总线接口+OpenMV 共用 UART2
  ps2.begin();
  carStop();
  lastPoll = lastPs2Poll = lastValidPs2 = millis();
  Serial.println("V1-1 READY + PS2 CAR CONTROL (bus motors)");
  Serial.print("PS2 pins: DAT="); Serial.print(PS2_DAT);
  Serial.print(" CMD="); Serial.print(PS2_CMD);
  Serial.print(" ATT="); Serial.print(PS2_ATT);
  Serial.print(" CLK="); Serial.println(PS2_CLK);
  Serial.print("CarBus: via board bus port (UART2 RX="); Serial.print(BUS_RX);
  Serial.print(" TX="); Serial.print(BUS_TX);
  Serial.print(" baud="); Serial.print(BAUD);
  Serial.print(" ch="); Serial.print(CAR_CH_LB);
  Serial.print("/"); Serial.print(CAR_CH_RB);
  Serial.print("/"); Serial.print(CAR_CH_LT);
  Serial.print("/"); Serial.print(CAR_CH_RT);
  Serial.print(" inv="); Serial.print(CAR_INV_LB ? 1 : 0);
  Serial.print(CAR_INV_RB ? 1 : 0);
  Serial.print(CAR_INV_LT ? 1 : 0);
  Serial.print(CAR_INV_RT ? 1 : 0);
  Serial.print(" braces="); Serial.println(CAR_BUS_BRACES ? 1 : 0);
  help();
}

void loop() {
  uint32_t now = millis();
  if (rxUsed && now - lastRx > 100) rxUsed = 0;
  if (pending && now - sentAt >= TIMEOUT_MS) {
    pending = false; ++timeouts;
    Serial.println("TIMEOUT: execution state unknown.");
  }
  // ---- 新增: PS2 轮询(20ms)与小车控制 ----
  static uint8_t ps2Streak = 0;   // 连续有效帧计数: 未配对/乱码时保持0
  if (now - lastPs2Poll >= PS2_POLL_MS) {
    lastPs2Poll = now;
    ps2.writeRead();
    if (ps2.dataReady()) { if (ps2Streak < 250) ++ps2Streak; lastValidPs2 = now; }
    else ps2Streak = 0;
  }
  // ---- 调试: 每 500ms 打印一次手柄帧/摇杆/按键 ----
  static uint32_t lastDbg = 0;
  if (now - lastDbg >= 500) {
    lastDbg = now;
    char tmp[32], mhex[4];
    snprintf(tmp, sizeof(tmp), "%02X %02X %02X %02X %02X %02X %02X %02X %02X",
             ps2.byte(0), ps2.byte(1), ps2.byte(2), ps2.byte(3), ps2.byte(4),
             ps2.byte(5), ps2.byte(6), ps2.byte(7), ps2.byte(8));
    snprintf(mhex, sizeof(mhex), "%02X", ps2.modeByte());
    Serial.print("PS2[");
    Serial.print(tmp);
    Serial.print("] mode=");
    Serial.print(mhex);
    if (ps2.dataReady()) {
      Serial.print(" ready s="); Serial.print(ps2Streak);
      Serial.print(" LY="); Serial.print(ps2.ly());
      Serial.print(" RY="); Serial.print(ps2.ry());
      Serial.print(" L3="); Serial.print(ps2.pressed(PS2_L3) ? 1 : 0);
      Serial.print(" R3="); Serial.print(ps2.pressed(PS2_R3) ? 1 : 0);
      Serial.print(" ST="); Serial.println(ps2.pressed(PS2_START) ? 1 : 0);
    } else {
      Serial.print(" NOFRAME(ms=");
      Serial.print(millis() - lastValidPs2);
      Serial.println(")");
    }
  }
  const bool handleOk = ps2Streak >= 3;  // 连续≥3帧(约60ms)有效才认, 防止未配对乱码驱动电机
  if (handleOk) {
    // SELECT 按下沿 -> 车<->云台 模式切换
    bool selPressed = ps2.pressed(PS2_SELECT);
    if (selPressed && !selWasPressed) {
      gimbalMode = (gimbalMode == MODE_CAR) ? MODE_GIMBAL : MODE_CAR;
      if (gimbalMode == MODE_GIMBAL) {
        carStop(); lastCarPL = lastCarPR = 1500;
        pendingGimbalCmd = "HOLD";   // 云台模式: 停止自动识别
        Serial.println("MODE: GIMBAL (manual JOG; auto OFF)");
      } else {
        pendingGimbalCmd = "TRACK";  // 车模式: 恢复自动识别
        Serial.println("MODE: CAR (gimbal auto-resume)");
      }
    }
    selWasPressed = selPressed;

    bool startPressed = ps2.pressed(PS2_START);
    if (startPressed && !startWasPressed) {
      if (gimbalMode == MODE_GIMBAL) {
        sendCommand("HOME");         // 云台模式: START=回中
      } else {
        carStop();                   // 车模式: START=停车
        lastCarPL = lastCarPR = 1500;
      }
    }
    startWasPressed = startPressed;

    if (gimbalMode == MODE_CAR) handleCar();
    else handleGimbal();
  } else {
    startWasPressed = true;
    selWasPressed = true;
    if (lastCarPL != 1500 || lastCarPR != 1500) {  // 未配对/失联: 发一次停车并保持
      carStop();
      lastCarPL = lastCarPR = 1500;
    }
  }
  // 模式切换的 HOLD/TRACK 待发指令: 总线空闲后发出(只发一次)
  if (pendingGimbalCmd && !pending) {
    sendCommand(pendingGimbalCmd);
    pendingGimbalCmd = nullptr;
  }
  // ----------------------------------------
  for (int n = 0; n < 256 && Bus.available(); ++n) receiveByte(static_cast<char>(Bus.read()));
  for (int n = 0; n < 128 && Serial.available(); ++n) {
    char value = static_cast<char>(Serial.read());
    if (value == '\r') continue;
    if (value == '\n') {
      if (!dropping) { input[inputUsed] = '\0'; command(input); }
      inputUsed = 0; dropping = false;
    } else if (!dropping) {
      if (inputUsed >= sizeof(input) - 1) dropping = true;
      else input[inputUsed++] = value;
    }
  }
  if (polling && !pending && millis() - lastPoll >= POLL_MS) {
    sendCommand(nextInfo ? "INFO" : "STATUS", nullptr, true);
    nextInfo = !nextInfo;
  }
  delay(2);
}
