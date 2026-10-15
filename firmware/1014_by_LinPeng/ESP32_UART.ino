/* ============================================================================
 * KPZ-ESP32 小车主控固件  V1-13
 * 平台: DOIT ESP32 DEVKIT V1 + KPZ-ESP32 扩展板 (Arduino core 3.3.x)
 * ----------------------------------------------------------------------------
 * 【功能总览】
 *   上电默认模式由 BOOT_MODE 决定, 现为 MODE_TRACK (上电即自动视觉跟踪);
 *   一把 PS2 手柄 + SELECT 键循环切换四种模式:
 *     1) CAR    手动车   : 手柄左右摇杆分别驱动左右电机 (TB6612 双路)
 *     2) GIMBAL 手动云台 : 手柄摇杆直接控制两个舵机 pan/tilt (ESP32 输出 PWM,
 *                          不再经过 OpenMV; 切到云台模式 = 手动接管, 关掉自动识别)
 *     3) LINE   自动巡线 : 8 路 I2C 循迹传感器自动循迹 + 停车标记检测,
 *                          停车后等待继续信号(DONE / go)再恢复行驶
 *     4) TRACK  视觉跟踪 : 云台由 ESP32 独占驱动(本机输出 PWM), OpenMV 只负责
 *                          "看": 它通过 UART2 上报
 *                              $G2,TGT,<cx>,<cy>,<lx>,<ly>,<aim>,<run>!
 *                          本机把像素误差换成微秒并做瞄准 PID, 输出 PAN=27/TILT=14
 *                          (瞄准模型移植自 OpenMV main(5).py 的速度/加速度限制)
 * ----------------------------------------------------------------------------
 * 【引脚表】(改线前先看这里, 以此为准)
 *   PS2 手柄接收器 : DAT=GPIO19  CMD=GPIO18  ATT=GPIO15  CLK=GPIO23
 *                    (对应接收器上 DI / DO / CS / CK, 注意排线顺序)
 *   TB6612 电机   : STBY 接扩展板 3V3 (常通, 不占 GPIO)
 *                    左电机: PWM=GPIO32   AIN1=GPIO13   AIN2=GPIO12
 *                    右电机: PWM=GPIO33   AIN1=GPIO2    AIN2=GPIO4
 *                    (AIN 只是方向电平, 用普通 GPIO; 转向反了改 TB_INV_L/R)
 *   云台舵机      : pan = GPIO27 (扩展板 PWM舵机接口 4 号, V/G 供 5V)
 *                    tilt= GPIO14 (扩展板 PWM舵机接口 5 号, V/G 供 5V)
 *                    注意: 舵机电源负极(GND) 必须与 ESP32 GND 共地
 *   循迹传感器    : I2C 地址 0x5D, SDA=GPIO21  SCL=GPIO22 (S6 口, VCC=5V)
 *                    读寄存器 5, 1 字节: bit0..7 = 通道1..8, bit=1 表示压到黑线
 *   OpenMV / UART2: RX=GPIO16  TX=GPIO17  (115200)
 *                    接线: ESP32 GPIO16(RX) <- OpenMV P4(TX)
 *                          ESP32 GPIO17(TX) -> OpenMV P5(RX),  GND 共地
 *                    收 $G2,TGT 视觉帧(见 TRACK 模式) 与 "PARKED" 裸串;
 *                    发 $G2 命令帧; 巡线停车后收 "DONE" 裸串恢复行驶
 * ----------------------------------------------------------------------------
 * 【手柄操作】
 *   SELECT : 循环切换模式  CAR -> GIMBAL -> LINE -> TRACK -> CAR
 *   --- CAR 手动车 ---
 *     左摇杆(上下) : 左电机速度       右摇杆(上下) : 右电机速度
 *     L3 / R3      : 原地左旋 / 右旋  START : 停车
 *   --- GIMBAL 手动云台 (增量微调, 松手保持当前角度) ---
 *     左摇杆(左右) : pan  水平转向    右摇杆(上下) : tilt 俯仰
 *     START        : 双舵机回中 (1500us)
 *   --- LINE 自动巡线 ---
 *     上电默认 CAR; 切到 LINE 后自动循迹
 *     START : 手动停车; 串口发 go 恢复
 *   --- TRACK 视觉跟踪 ---
 *     切入后本机给 OpenMV 发 $G2,TRACK, OpenMV 开始回 $G2,TGT;
 *     本机按像素误差做瞄准 PID 驱动 pan/tilt; 目标丢失/超时则原地保持
 *     START : 停住云台 (发 $G2,HOLD)
 * ----------------------------------------------------------------------------
 * 【串口命令速查】(USB 串口 115200, 回车发送)
 *   $G2 组: start / track / stop / hold / off / home / calpan
 *           ping / status / info / limits / stats / pollon / polloff
 *   视觉:   vision = 进入 TRACK 模式 (ESP32 独占云台, 消费 OpenMV 的 $G2,TGT)
 *           aim    = 打印瞄准状态      tgt = 打印最近一帧视觉数据
 *           pan+ / pan- / tilt+ / tilt- = 本地微调舵机 (不再是发给 OpenMV)
 *   电机:   cf 前进1s / cb 后退1s / cs 停止 / ch <0左|1右> <速度-255~255> <时长ms>
 *   舵机:   s1 <us> = pan, s2 <us> = tilt, sh = 回中
 *   巡线:   line 进入 / manual 退出 / 1|2|3 = 低速|中速|高速
 *           kp/ki/kd <值> 调 PID / go 恢复 / park 模拟停车 / s 查状态
 * ----------------------------------------------------------------------------
 * 【快速上手测试顺序】
 *   1. 上电看串口 "V1-13 READY" 与引脚打印
 *   2. 舵机测试: s1 1500 / s2 1500 (回中), 切 GIMBAL 推摇杆看 pan/tilt
 *   3. 电机测试: ch 0 200 500 / ch 1 200 500 (左右轮各转 0.5s), 方向反改 TB_INV
 *   4. 手柄:     先开手柄再上电, 看串口 PS2[...] ready 帧出现
 *   5. 巡线:     串口发 line, 放线上看 [LINE] S=xxxxxxxx i2c=ok
 *   6. 视觉链路: 先确认 OpenMV 已烧 main(6).py, 串口发 ping 应看到 REPLY $G2,PONG;
 *                再发 vision 进入 TRACK, 观察 [AIM] 帧与 $G2,TGT 是否持续刷新
 * ----------------------------------------------------------------------------
 * 版本记录: V1-6 电机按接线说明 + 巡线合入
 *           V1-7 AIN 改普通 GPIO(13/12/2/4), 舵机保持 27/14;
 *                云台控制交换: 左摇杆X=pan, 右摇杆Y=tilt
 *           V1-8 新增 TRACK 模式: OpenMV 只上报 $G2,TGT 像素坐标, 瞄准 PID 移到
 *                本机驱动 27/14; 新增 TGT 解析/目标超时保护/本地 jog; SELECT 四态
 *           V1-9 按实机测得的图像增益(pan≈3.0 px/us, tilt≈2.1 px/us)下调限速:
 *                pan 800->60us/s, tilt 200->60us/s, 加速度 2400->400us/s^2;
 *                旧值单周期位移达 48px(死区 4 倍)导致明显过冲来回甩
 *           V1-10 死区 12 -> 120px: 实测舵机最小可分辨步长约 40-60us
 *                (合 120~180px), 12px 死区相当于 4us 指令, 硬件执行不了,
 *                闭环会形成 ~6s 周期的极限环; 放大死区后进入范围即停住
 *           V1-11 TRACK 自愈: 只要还没收到 run=1, 每 2s 重发一次 $G2,TRACK.
 *                (OpenMV 停止脚本后看门狗会复位一次, 那瞬间的命令会丢)
 *           V1-12 新增 BOOT_MODE, 默认 MODE_TRACK: 上电/复位即自动跟踪,
 *                配合 OpenMV 的 AUTO_START, 两端都能自愈
 *           V1-13 高速目标: TGT 帧扩展为 8 字段(带 vx,vy), 新增速度前馈
 *                + 60ms 预瞄; 限速 60->800/300, 加速度 400->6000;
 *                移动中的目标不再被判为"已对准"(否则会停在原地被甩掉)
 * ============================================================================
 */
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <string.h>

// ===================== 配置区（按实际接线修改） =====================
// ---- PS2 手柄接收器引脚（DAT/CMD/ATT/CLK 对应接收器上 DI/DO/CS/CK）----
constexpr uint8_t PS2_DAT = 19, PS2_CMD = 18, PS2_ATT = 15, PS2_CLK = 23;

// ---- 小车 TB6612 驱动（AIN 用普通 GPIO, 不占 PWM 舵机接口; STBY 接 3V3 常通）----
constexpr uint8_t PIN_PWM_L = 32;   // 左电机 PWM (PWM0)
constexpr uint8_t PIN_AIN1_L = 13;  // 左 AIN1 (S4 接口)
constexpr uint8_t PIN_AIN2_L = 12;  // 左 AIN2 (S4 接口)
constexpr uint8_t PIN_PWM_R = 33;   // 右电机 PWM (PWM1)
constexpr uint8_t PIN_AIN1_R = 2;   // 右 AIN1 (板上普通排针)
constexpr uint8_t PIN_AIN2_R = 4;   // 右 AIN2 (板上普通排针)
constexpr int    TB_PWM_FREQ = 1000;   // PWM 频率 Hz
constexpr int    TB_PWM_RES = 8;       // 8bit -> 0~255
constexpr bool   TB_INV_L = false;     // 左方向反了改 true
constexpr bool   TB_INV_R = false;     // 右方向反了改 true
constexpr int16_t CAR_MAX_SPEED = 255;
constexpr int16_t CAR_SPEED_DEADZONE = 30;
constexpr int16_t CAR_SPIN_SPEED = 180;
constexpr uint32_t PS2_POLL_MS = 20;

// ---- 云台舵机（ESP32 直接 PWM 输出; 保持在扩展板 PWM 接口 4/5 号, 不动）----
constexpr uint8_t PIN_PAN = 27;    // pan(水平) 舵机信号 -> PWM舵机接口 4 号
constexpr uint8_t PIN_TILT = 14;   // tilt(俯仰) 舵机信号 -> PWM舵机接口 5 号
constexpr int    SERVO_FREQ = 50;
constexpr int    SERVO_RES = 16;
constexpr int    SERVO_MIN_US = 500;
constexpr int    SERVO_MAX_US = 2500;
constexpr int    SERVO_MID_US = 1500;
constexpr int16_t GIMBAL_DEADZONE = 20;
constexpr int16_t GIMBAL_STEP_DIV = 4;   // 步进: 满偏≈310us/s (想更慢改 6/8)
constexpr uint32_t GIMBAL_POLL_MS = 100;

// ---- 8 路循迹传感器（I2C 模式, S6 口 SDA21/SCL22, 5V 供电）----
constexpr uint8_t  I2C_SDA = 21, I2C_SCL = 22;
constexpr uint8_t  I2C_ADDR = 0x5D;
constexpr uint8_t  I2C_REG_STATE = 5;
constexpr uint32_t I2C_FREQ = 400000;
constexpr bool     SENSOR_FLIP = false;  // 通道1..8 与左→右相反时改 true

// ---- 巡线/停车参数（移植自 ESP32_LineCar.ino）----
constexpr int     PARKING_MIN_BLACK = 6;    // 同时压黑 >= 该路数判定停车标记
constexpr uint32_t PARKING_DEBOUNCE_MS = 80;
constexpr float   KP_DEFAULT = 2.0f, KI_DEFAULT = 0.0f, KD_DEFAULT = 1.2f;
constexpr int     SPEED_LOW = 110, SPEED_MED = 160, SPEED_HIGH = 210;
constexpr uint32_t CLEAR_MARKER_MIN_MS = 150;   // 越线最短直行
constexpr uint32_t CLEAR_MARKER_MAX_MS = 2000;  // 越线最长直行(超时强制恢复)
constexpr uint32_t PARK_BLANK_MS = 300;         // 恢复后停车检测空白期
constexpr uint32_t LOST_TIMEOUT_MS = 800;       // 丢线超时停车保护
constexpr float   LOST_SPEED_FACTOR = 0.6f;
constexpr uint32_t LINE_CTRL_MS = 10;           // 巡线控制周期 100Hz
constexpr uint32_t LINE_DEBUG_MS = 200;
const char CMD_DONE[] = "DONE";      // UART2 收到 DONE = 打靶完成, 继续行驶
const char RSP_PARKED[] = "PARKED";  // 停车后发送, 表示已停稳

// ---- UART2 ($G2 + DONE) ----
constexpr int BUS_RX = 16, BUS_TX = 17;
constexpr uint32_t BAUD = 115200, TIMEOUT_MS = 1000, POLL_MS = 2000;

// ---- 视觉瞄准 (V1-8 TRACK 模式) ----
// OpenMV 以 <=50Hz 上报 $G2,TGT,<cx>,<cy>,<lx>,<ly>,<aim>,<run>!
// 本机把像素误差换算成舵机微秒的变化率. 运动模型(增益/限速/加速度)移植自
// OpenMV main(5).py.
//
// V1-9 限速调参依据 —— 实机实测这台云台的"图像增益":
//     pan  ≈ 3.0 px/us  (点动 +60us -> 画面横移约 180px)
//     tilt ≈ 2.1 px/us  (点动 -30us -> 画面纵移约 60px)
//   控制周期 20ms、死区 12px. 限速过大时单周期画面位移会超过死区 -> 必然过冲:
//     旧值 pan  800us/s -> 3.0*800*0.02 = 48   px/步 (死区 4 倍, 实测来回甩)
//     旧值 tilt 200us/s -> 2.1*200*0.02 = 8.4  px/步 (临界)
//   现值 60/60us/s    -> 3.6 / 2.5 px/步, 稳定.
//   想要更快可上调, 但建议单步位移 <= 死区一半(6px): pan <= ~100, tilt <= ~140 us/s.
//
// V1-9 死区为什么放大到 120px —— 实测这台云台的舵机"最小可分辨步长":
//     点动 pan +20us -> 画面 0.0 px; +40us -> 0.0 px; +60us -> 突然跳 73~180px
//   即舵机对约 40-60us 以内的指令基本不响应, 一响应就跳 120~180px.
//   而 12px 死区只相当于 ~4us 的 pan 指令 —— 闭环一直在下达硬件执行不了的微指令,
//   攒够后突然跳一大步(死区的 10 倍), 于是永远在来回找.
//   死区取 120px 后, 一进入范围就停住. 想更准只能从机械上提高分辨率.
constexpr uint32_t AIM_CTRL_MS = 20;             // 瞄准控制周期 50Hz
constexpr uint32_t AIM_TARGET_TIMEOUT_MS = 500;  // 超过这么久没 TGT -> 原地保持
constexpr int      AIM_CENTER_X = 320, AIM_CENTER_Y = 240;  // VGA 图像中心
constexpr int      AIM_DEADBAND_PX = 120;  // 未对准时的死区 (V1-9: 12 -> 120)
constexpr int      AIM_RESTART_PX = 150;   // 已对准后的保持死区(防抖)
constexpr float    AIM_GAIN_NEAR = 1.8f;   // |误差| <= 64px  (us/s per px)
constexpr float    AIM_GAIN_MID = 2.8f;    // |误差| <= 120px
constexpr float    AIM_GAIN_FAR = 4.0f;    // |误差| >  120px
constexpr float    AIM_TILT_GAIN_CAP = 1.2f;
// ---- V1-13 高速目标: 前馈 + 预瞄 + 重新放开限速 ----
// 实测图像增益: 1us 舵机行程 -> pan 约 3.0px / tilt 约 2.1px 的画面位移.
// 目标速度(px/s) -> 需要的舵机速度(us/s) = v / PX_PER_US, 这就是前馈项.
constexpr float    AIM_PAN_PX_PER_US  = 3.0f;
constexpr float    AIM_TILT_PX_PER_US = 2.1f;
constexpr float    AIM_LOOKAHEAD_MS = 60.0f;   // 预瞄: 补偿视觉+串口+舵机延迟
constexpr float    AIM_MOVING_PX_S  = 30.0f;   // 速度超过它就不判定为"已对准"
// V1-9 把限速压到 60us/s 是因为当时死区只有 12px; 死区现在 120px, 稳定性判据
// 变成"单步位移 <= 死区": 120px/20ms/3.0us = 2000us/s, 所以能大幅放开.
//   pan  800us/s -> 3.0*800*0.02 = 48   px/步 (死区 120px 的 40%)
//   tilt 300us/s -> 2.1*300*0.02 = 12.6 px/步
constexpr float    AIM_PAN_MAX_US_S = 800.0f;    // V1-9 曾降到 60
constexpr float    AIM_TILT_MAX_US_S = 300.0f;   // V1-9 曾降到 60
constexpr float    AIM_ACCEL_US_S2 = 6000.0f;    // V1-9 曾降到 400
constexpr int8_t   AIM_PAN_RESPONSE = 1;   // 转反了改 -1
constexpr int8_t   AIM_TILT_RESPONSE = 1;  // 转反了改 -1
// 机械行程(与 OpenMV 板上 v1_tracking_limits.json 记录的一致)
constexpr int      AIM_PAN_MIN_US = 740, AIM_PAN_MAX_US = 2260;
constexpr int      AIM_TILT_MIN_US = 1200, AIM_TILT_MAX_US = 1800;
// ================================================================

// PS2 按键位（bit=0 表示按下, active-low）
enum : uint16_t {
  PS2_L2 = 1 << 0, PS2_R2 = 1 << 1, PS2_L1 = 1 << 2, PS2_R1 = 1 << 3,
  PS2_TRI = 1 << 4, PS2_CIRC = 1 << 5, PS2_CROSS = 1 << 6, PS2_SQUARE = 1 << 7,
  PS2_SELECT = 1 << 8, PS2_L3 = 1 << 9, PS2_R3 = 1 << 10, PS2_START = 1 << 11,
  PS2_UP = 1 << 12, PS2_RIGHT = 1 << 13, PS2_DOWN = 1 << 14, PS2_LEFT = 1 << 15
};

// ---------- PS2 位操作驱动（移植自 Carbot 出厂代码 z_ps2.py） ----------
class PS2Controller {
 public:
  PS2Controller(uint8_t dat, uint8_t cmd, uint8_t att, uint8_t clk)
      : dat_(dat), cmd_(cmd), att_(att), clk_(clk) {
    memset(buf_, 0, sizeof(buf_));
  }
  void begin() {
    pinMode(dat_, INPUT_PULLUP);
    pinMode(cmd_, OUTPUT); pinMode(att_, OUTPUT); pinMode(clk_, OUTPUT);
    digitalWrite(cmd_, HIGH); digitalWrite(clk_, HIGH); digitalWrite(att_, HIGH);
  }
  void writeRead() {
    digitalWrite(att_, LOW);
    buf_[0] = transfer(0x01);
    buf_[1] = transfer(0x42);
    for (int i = 2; i < 9; ++i) buf_[i] = transfer(buf_[0]);
    digitalWrite(att_, HIGH);
  }
  bool dataReady() const { return buf_[0] == 0xFF && buf_[1] == 0x73 && buf_[2] == 0x5A; }
  uint8_t modeByte() const { return buf_[1]; }
  uint8_t byte(size_t i) const { return (i < sizeof(buf_)) ? buf_[i] : 0; }
  uint16_t buttons() const { return (uint16_t(buf_[3]) << 8) | buf_[4]; }
  bool pressed(uint16_t mask) const { return (buttons() & mask) == 0; }
  int16_t lx() const { return buf_[7]; }
  int16_t ly() const { return buf_[8]; }
  int16_t rx() const { return buf_[5]; }
  int16_t ry() const { return buf_[6]; }
 private:
  uint8_t transfer(uint8_t data) {
    uint8_t readData = 0;
    for (uint8_t i = 0; i < 8; ++i) {
      digitalWrite(cmd_, (data & (0x01 << i)) ? HIGH : LOW);
      digitalWrite(clk_, HIGH); delayMicroseconds(5);
      digitalWrite(clk_, LOW); delayMicroseconds(5);
      if (digitalRead(dat_) == HIGH) readData |= 0x01 << i;
    }
    digitalWrite(clk_, HIGH); delayMicroseconds(10);
    return readData;
  }
  uint8_t dat_, cmd_, att_, clk_;
  uint8_t buf_[9];
};

// ---------- 驱动/舵机前置声明（供串口命令调用） ----------
void tbDrive(int16_t lSpeed, int16_t rSpeed);
void carStop();
void tbBrake();
void help();
int16_t lastCarLS = 0, lastCarRS = 0;
void setServoUs(uint8_t pin, int us);
int16_t panUs = SERVO_MID_US, tiltUs = SERVO_MID_US;

// Arduino 构建系统会把自动生成的原型插到"第一个函数定义"之前, 而 struct PID 要到
// 巡线段才完整定义; 不先声明的话, 生成的 "float lnPidCompute(PID&, float, float);"
// 原型会因为 PID 尚未声明而编译失败 (实测 arduino-cli + esp32 3.3.12 必现, 原
// V1-7 结构同样如此). 引用不完整类型对"声明"是合法的, 所以这一行就够了.
struct PID;

// ---------- $G2 协议与控制台（原逻辑保留; 轮询默认关） ----------
HardwareSerial Bus(2);
char rx[192] = {}, input[40] = {}, expected[48] = {}, errorPrefix[32] = {};
size_t rxUsed = 0, inputUsed = 0;
bool dropping = false, pending = false, polling = false, nextInfo = true;
bool pendingPoll = false;
uint16_t sequence = 0;
uint32_t sentAt = 0, lastRx = 0, lastPoll = 0;
uint32_t requests = 0, replies = 0, timeouts = 0, writeErrors = 0;
uint32_t laserHitCount = 0;

// ---------- $G2,TGT 视觉瞄准状态 (V1-8) ----------
int      aimCx = -1, aimCy = -1, aimLx = -1, aimLy = -1;
int      aimVx = 0, aimVy = 0;         // target velocity px/s, from the OpenMV
int      aimMode = 0;          // 0 = 目标对屏幕中心, 1 = 把激光压到目标上
bool     aimRun = false;       // 0/1 from the OpenMV: may we move?
uint32_t aimLastFrameMs = 0, aimLastCtrlMs = 0, aimLastTrackCmd = 0;
uint32_t aimFrames = 0, aimBadFrames = 0;
float    aimPosUs[2] = {float(SERVO_MID_US), float(SERVO_MID_US)};
float    aimVelUs[2] = {0.0f, 0.0f};
bool     aimCentered[2] = {false, false};
bool     aimAtLimit = false;
const char *aimReason = "IDLE";
void aimSyncFromServo();
void aimControlUpdate(uint32_t now);
void aimJog(uint8_t axis, int deltaUs);
void parseTargetFrame(const char *frame);
void aimPrint();

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

void      receiveByte(char value) {
  lastRx = millis();
  if (value == '$') { rxUsed = 0; rx[rxUsed++] = value; }
  else if (rxUsed) {
    if (rxUsed >= sizeof(rx) - 1) { rxUsed = 0; return; }
    rx[rxUsed++] = value;
    if (value == '!') {
      rx[rxUsed] = '\0';
      // 视觉数据帧: 不属于命令-应答配对, 直接吃掉并更新瞄准状态.
      if (strncmp(rx, "$G2,TGT,", 8) == 0) {
        parseTargetFrame(rx);
        rxUsed = 0;
        return;
      }
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

// ---------- 巡线状态机（移植自 ESP32_LineCar.ino） ----------
enum LineState { LS_FOLLOW, LS_PARKING, LS_CLEAR, LS_IDLE };
LineState lnState = LS_FOLLOW;
unsigned long lnStateEnterMs = 0, lnParkBlankUntilMs = 0;

struct PID { float kp, ki, kd; float integral, prevError, integralMax; bool hasPrev; };
PID lnPid = { KP_DEFAULT, KI_DEFAULT, KD_DEFAULT, 0, 0, 30.0f, false };
int lnBaseSpeed = SPEED_MED;

uint8_t lnSensorBits = 0;
bool    lnSensorOk = false;
int     lnBlackCount = 0;
float   lnPosition = 3.5f;
float   lnError = 0;
int     lnLSpeed = 0, lnRSpeed = 0;
bool    lnLineLost = false;
unsigned long lnLostStartMs = 0;
unsigned long lnLastControlMs = 0, lnLastDebugMs = 0;

void lnResetPID() {
  lnPid.integral = 0; lnPid.prevError = 0; lnPid.hasPrev = false;
}

void lnReadLineSensors() {
  lnSensorBits = 0; lnBlackCount = 0;
  Wire.beginTransmission(I2C_ADDR);
  Wire.write(I2C_REG_STATE);
  if (Wire.endTransmission() != 0) {
    lnSensorOk = false; lnPosition = 3.5f; return;
  }
  if (Wire.requestFrom(I2C_ADDR, 1) != 1) {
    lnSensorOk = false; lnPosition = 3.5f; return;
  }
  lnSensorOk = true;
  lnSensorBits = (uint8_t)Wire.read();
  int sum = 0;
  for (int i = 0; i < 8; i++) {
    if (lnSensorBits & (1 << i)) {
      int idx = SENSOR_FLIP ? (7 - i) : i;
      sum += idx;
      lnBlackCount++;
    }
  }
  lnPosition = (lnBlackCount > 0) ? ((float)sum / lnBlackCount) : 3.5f;
}

float lnPidCompute(PID &p, float error, float dt) {
  if (!p.hasPrev) {
    p.hasPrev = true; p.prevError = error; p.integral = 0;
    return p.kp * error;
  }
  p.integral += error * dt;
  if (p.integral > p.integralMax) p.integral = p.integralMax;
  else if (p.integral < -p.integralMax) p.integral = -p.integralMax;
  float derivative = (error - p.prevError) / dt;
  p.prevError = error;
  return p.kp * error + p.ki * p.integral + p.kd * derivative;
}

void lnEnterParking() {
  lnState = LS_PARKING;
  lnStateEnterMs = millis();
  tbBrake();
  lnResetPID();
  Serial.println(RSP_PARKED);
  Bus.println(RSP_PARKED);   // 上报车已停稳, 等待 DONE
  Serial.println("检测到停车标记 -> 停车, 等待继续信号");
}

void lnResumeFromParking() {
  lnState = LS_CLEAR;
  lnStateEnterMs = millis();
  lnResetPID();
  Serial.println("收到继续信号 -> 越线恢复");
}

void lnHandleParkingDetection() {
  if (millis() < lnParkBlankUntilMs) return;
  static bool pending = false;
  static unsigned long pendingStart = 0;
  if (lnBlackCount >= PARKING_MIN_BLACK) {
    if (!pending) { pending = true; pendingStart = millis(); }
    else if (millis() - pendingStart >= PARKING_DEBOUNCE_MS) {
      pending = false;
      lnEnterParking();
    }
  } else {
    pending = false;
  }
}

void lnFollowLine(float dt) {
  float error = (lnPosition - 3.5f) / 3.5f;
  int speed = lnBaseSpeed;
  float correction;
  if (lnBlackCount == 0) {
    if (!lnLineLost) { lnLineLost = true; lnLostStartMs = millis(); lnResetPID(); }
    if (millis() - lnLostStartMs > LOST_TIMEOUT_MS) {
      tbBrake();
      Serial.println("丢线超时 -> 停车保护");
      return;
    }
    speed = (int)(lnBaseSpeed * LOST_SPEED_FACTOR);
    error = lnError;
    correction = lnPid.kp * error;
  } else {
    lnLineLost = false;
    correction = lnPidCompute(lnPid, error, dt);
  }
  lnError = error;
  lnLSpeed = speed + (int)correction;
  lnRSpeed = speed - (int)correction;
  tbDrive(lnLSpeed, lnRSpeed);
}

void lnRunControl(float dt) {
  lnReadLineSensors();
  switch (lnState) {
    case LS_FOLLOW:
      lnHandleParkingDetection();
      if (lnState == LS_FOLLOW) lnFollowLine(dt);
      break;
    case LS_PARKING:
      break;   // 保持刹车, 等 DONE / go
    case LS_CLEAR: {
      unsigned long elapsed = millis() - lnStateEnterMs;
      bool cleared = (lnBlackCount < PARKING_MIN_BLACK) && (elapsed >= CLEAR_MARKER_MIN_MS);
      bool timedOut = (elapsed >= CLEAR_MARKER_MAX_MS);
      if (cleared || timedOut) {
        lnState = LS_FOLLOW;
        lnResetPID();
        lnParkBlankUntilMs = millis() + PARK_BLANK_MS;
        Serial.println(timedOut ? "越线超时 -> 强制恢复" : "越线完成 -> 恢复循迹");
      } else {
        tbDrive(lnBaseSpeed, lnBaseSpeed);
      }
      break;
    }
    case LS_IDLE:
      tbDrive(0, 0);
      break;
  }
}

// ---------- 模式（四态: 手动车 / 云台 / 巡线 / 视觉跟踪） ----------
enum : uint8_t { MODE_CAR = 0, MODE_GIMBAL = 1, MODE_LINE = 2, MODE_TRACK = 3 };
// 上电/复位后默认进入哪个模式.
//   MODE_TRACK = 上电即自动视觉跟踪 (配合 OpenMV 的 AUTO_START, 两边都能自愈:
//                OpenMV 被看门狗复位后自己恢复跟踪, ESP32 复位后自己进 TRACK)
//   MODE_CAR   = 上电云台保持不动, 需要串口敲 vision 才跟踪 (更保守)
constexpr uint8_t BOOT_MODE = MODE_TRACK;
uint8_t gimbalMode = MODE_CAR;
bool selWasPressed = true;
uint32_t lastGimbalSend = 0;
PS2Controller ps2(PS2_DAT, PS2_CMD, PS2_ATT, PS2_CLK);
uint32_t lastPs2Poll = 0, lastValidPs2 = 0;
bool startWasPressed = true;

void switchMode(uint8_t newMode) {
  const uint8_t previous = gimbalMode;
  gimbalMode = newMode;
  carStop(); lastCarLS = lastCarRS = 0;
  if (previous == MODE_TRACK && newMode != MODE_TRACK) {
    aimRun = false;                       // 离开跟踪: 冻结, 并让 OpenMV 停发
    aimVelUs[0] = aimVelUs[1] = 0.0f;
    aimReason = "HOLD";
    sendCommand("HOLD");
  }
  if (newMode == MODE_GIMBAL) {
    Serial.println("MODE: GIMBAL (manual servo PWM)");
  } else if (newMode == MODE_LINE) {
    lnState = LS_FOLLOW; lnResetPID();
    Serial.println("MODE: LINE (auto follow)");
  } else if (newMode == MODE_TRACK) {
    aimSyncFromServo();                   // 从当前舵机角度起步, 不跳变
    aimRun = false;
    aimLastFrameMs = millis();
    aimLastTrackCmd = millis();
    aimReason = "WAIT_TGT";
    Serial.println("MODE: TRACK (vision aim from OpenMV $G2,TGT)");
    sendCommand("TRACK");                 // 让 OpenMV 开始跟踪并上报
  } else {
    Serial.println("MODE: CAR (manual TB6612)");
  }
}

// ---------- 视觉瞄准控制 (V1-8) ----------
void aimSyncFromServo() {
  aimPosUs[0] = float(panUs);
  aimPosUs[1] = float(tiltUs);
  aimVelUs[0] = aimVelUs[1] = 0.0f;
  aimCentered[0] = aimCentered[1] = false;
  aimAtLimit = false;
}

void aimJog(uint8_t axis, int deltaUs) {   // 本地微调, 不再走总线
  const int low  = (axis == 0) ? AIM_PAN_MIN_US  : AIM_TILT_MIN_US;
  const int high = (axis == 0) ? AIM_PAN_MAX_US  : AIM_TILT_MAX_US;
  int value = constrain((axis == 0 ? int(panUs) : int(tiltUs)) + deltaUs, low, high);
  aimPosUs[axis] = float(value);
  aimVelUs[axis] = 0.0f;
  if (axis == 0) { panUs  = int16_t(value); setServoUs(PIN_PAN,  value); }
  else           { tiltUs = int16_t(value); setServoUs(PIN_TILT, value); }
}

void parseTargetFrame(const char *frame) {
  // "$G2,TGT,<cx>,<cy>,<vx>,<vy>,<lx>,<ly>,<aim>,<run>!"  (V6 OpenMV, frame+8
  // skips "$G2,TGT,").  vx/vy are the target velocity in px/s.
  int cx, cy, vx, vy, lx, ly, mode, run;
  if (sscanf(frame + 8, "%d,%d,%d,%d,%d,%d,%d,%d",
             &cx, &cy, &vx, &vy, &lx, &ly, &mode, &run) != 8) {
    ++aimBadFrames;
    return;
  }
  aimCx = cx; aimCy = cy; aimLx = lx; aimLy = ly;
  aimVx = vx; aimVy = vy;
  aimMode = (mode == 1) ? 1 : 0;
  aimRun = (run == 1);
  aimLastFrameMs = millis();
  ++aimFrames;
}

void aimControlUpdate(uint32_t now) {
  if (gimbalMode != MODE_TRACK) return;
  if (now - aimLastCtrlMs < AIM_CTRL_MS) return;
  const float dt = AIM_CTRL_MS / 1000.0f;
  aimLastCtrlMs = now;

  // 看门狗: 没有数据 / 目标丢失 / OpenMV 说别动 -> 原地冻结, 绝不自己乱跑
  if (!aimRun || aimCx < 0 || (now - aimLastFrameMs) > AIM_TARGET_TIMEOUT_MS) {
    aimVelUs[0] = aimVelUs[1] = 0.0f;
    aimReason = !aimRun ? "HOLD" : (aimCx < 0 ? "NO_TARGET" : "STALE");
    return;
  }
  const int refX = (aimMode == 1 && aimLx >= 0) ? aimLx : AIM_CENTER_X;
  const int refY = (aimMode == 1 && aimLy >= 0) ? aimLy : AIM_CENTER_Y;
  const int8_t resp[2] = {AIM_PAN_RESPONSE, AIM_TILT_RESPONSE};
  // Target velocity -> feed-forward rate.  The servo rate that exactly cancels
  // the target's image motion is -resp * v / PX_PER_US; adding it means the
  // loop MOVES WITH the target instead of永远 chasing the last measurement.
  const float vel[2]  = {float(aimVx), float(aimVy)};
  const float px_us[2] = {AIM_PAN_PX_PER_US, AIM_TILT_PX_PER_US};
  // Look-ahead: by the time this command reaches the servo the target has moved
  // on, so aim where it WILL be, not where it was.
  const float lead = AIM_LOOKAHEAD_MS / 1000.0f;
  const float err[2] = {float(aimCx - refX) + vel[0] * lead,
                        float(aimCy - refY) + vel[1] * lead};
  const float ff[2] = {-float(resp[0]) * vel[0] / px_us[0],
                       -float(resp[1]) * vel[1] / px_us[1]};
  aimAtLimit = false;
  for (uint8_t axis = 0; axis < 2; ++axis) {
    const int band = aimCentered[axis] ? AIM_RESTART_PX : AIM_DEADBAND_PX;
    // Only treat "inside the deadband" as done when the target is essentially
    // stationary; a moving target must keep being followed or we fall behind.
    if (fabsf(err[axis]) <= float(band) &&
        fabsf(vel[axis]) < AIM_MOVING_PX_S) {
      aimCentered[axis] = true;
      aimVelUs[axis] = 0.0f;
      continue;
    }
    aimCentered[axis] = false;
    aimReason = "AIMING";
    const float magnitude = fabsf(err[axis]);
    float gain = (magnitude <= 64.0f) ? AIM_GAIN_NEAR
               : (magnitude <= 120.0f) ? AIM_GAIN_MID : AIM_GAIN_FAR;
    if (axis == 1 && gain > AIM_TILT_GAIN_CAP) gain = AIM_TILT_GAIN_CAP;
    const float cap = (axis == 0) ? AIM_PAN_MAX_US_S : AIM_TILT_MAX_US_S;
    float desired = constrain(-float(resp[axis]) * gain * err[axis] + ff[axis],
                              -cap, cap);
    float start = aimVelUs[axis];
    // Do not brake against the feed-forward: only zero the integrator-like ramp
    // when the command genuinely reverses direction.
    if (start * desired <= 0.0f) start = 0.0f;               // 反向先刹停
    else if (fabsf(desired) < fabsf(start)) start = desired; // 减速直接取目标
    const float velocity =
        fmaxf(start - AIM_ACCEL_US_S2 * dt,
              fminf(start + AIM_ACCEL_US_S2 * dt, desired));
    aimVelUs[axis] = velocity;
    const int low  = (axis == 0) ? AIM_PAN_MIN_US  : AIM_TILT_MIN_US;
    const int high = (axis == 0) ? AIM_PAN_MAX_US : AIM_TILT_MAX_US;
    float target = aimPosUs[axis] + velocity * dt;
    if (target < float(low) || target > float(high)) aimAtLimit = true;
    target = constrain(target, float(low), float(high));
    aimPosUs[axis] = target;
    if (axis == 0) { panUs  = int16_t(lroundf(target)); setServoUs(PIN_PAN,  panUs); }
    else           { tiltUs = int16_t(lroundf(target)); setServoUs(PIN_TILT, tiltUs); }
  }
  if (aimAtLimit) aimReason = "LIMIT";
  else if (aimCentered[0] && aimCentered[1]) aimReason = "ALIGNED";
}

void aimPrint() {
  Serial.print("AIM mode="); Serial.print(aimMode ? "LASER" : "CENTER");
  Serial.print(" run="); Serial.print(aimRun ? 1 : 0);
  Serial.print(" state="); Serial.print(aimReason);
  Serial.print(" tgt="); Serial.print(aimCx); Serial.print(","); Serial.print(aimCy);
  Serial.print(" laser="); Serial.print(aimLx); Serial.print(","); Serial.print(aimLy);
  Serial.print(" ref="); Serial.print(aimMode ? aimLx : AIM_CENTER_X);
  Serial.print(","); Serial.print(aimMode ? aimLy : AIM_CENTER_Y);
  Serial.print(" pwm="); Serial.print(panUs); Serial.print(","); Serial.print(tiltUs);
  Serial.print(" age_ms="); Serial.println(millis() - aimLastFrameMs);
}

// ---------- 舵机 PWM 直驱 ----------
void setServoUs(uint8_t pin, int us) {
  us = constrain(us, SERVO_MIN_US, SERVO_MAX_US);
  uint32_t duty = uint32_t(us) * 65535UL / 20000UL;
  ledcWrite(pin, duty);
}

void handleGimbal() {
  if (millis() - lastGimbalSend < GIMBAL_POLL_MS) return;
  lastGimbalSend = millis();
  int16_t panD = ps2.lx() - 128;
  int16_t tiltD = 128 - ps2.ry();
  if (abs(panD) > GIMBAL_DEADZONE)  panUs  += panD  / GIMBAL_STEP_DIV;
  if (abs(tiltD) > GIMBAL_DEADZONE) tiltUs += tiltD / GIMBAL_STEP_DIV;
  setServoUs(PIN_PAN, panUs);
  setServoUs(PIN_TILT, tiltUs);
}

// ---------- TB6612 驱动 ----------
void tbSetSide(uint8_t ain1, uint8_t ain2, uint8_t pwmPin, int16_t speed) {
  if (speed > 0)      { digitalWrite(ain1, HIGH); digitalWrite(ain2, LOW);  }
  else if (speed < 0) { digitalWrite(ain1, LOW);  digitalWrite(ain2, HIGH); }
  else                { digitalWrite(ain1, LOW);  digitalWrite(ain2, LOW);  }
  ledcWrite(pwmPin, (speed < 0) ? -speed : speed);
}

void tbDrive(int16_t lSpeed, int16_t rSpeed) {
  if (TB_INV_L) lSpeed = -lSpeed;
  if (TB_INV_R) rSpeed = -rSpeed;
  lSpeed = constrain(lSpeed, -255, 255);
  rSpeed = constrain(rSpeed, -255, 255);
  tbSetSide(PIN_AIN1_L, PIN_AIN2_L, PIN_PWM_L, lSpeed);
  tbSetSide(PIN_AIN1_R, PIN_AIN2_R, PIN_PWM_R, rSpeed);
}

void tbBrake() {   // 短路刹车: IN1=IN2=HIGH + PWM 拉满
  digitalWrite(PIN_AIN1_L, HIGH); digitalWrite(PIN_AIN2_L, HIGH);
  digitalWrite(PIN_AIN1_R, HIGH); digitalWrite(PIN_AIN2_R, HIGH);
  ledcWrite(PIN_PWM_L, 255); ledcWrite(PIN_PWM_R, 255);
}

void carStop() { tbDrive(0, 0); }

int16_t stickToSpeed(int16_t stickY) { return int16_t((128 - stickY) * 2.0f); }

void handleCar() {
  if (ps2.pressed(PS2_L3)) {
    if (lastCarLS != -CAR_SPIN_SPEED || lastCarRS != CAR_SPIN_SPEED) {
      lastCarLS = -CAR_SPIN_SPEED; lastCarRS = CAR_SPIN_SPEED;
      tbDrive(lastCarLS, lastCarRS);
    }
    return;
  }
  if (ps2.pressed(PS2_R3)) {
    if (lastCarLS != CAR_SPIN_SPEED || lastCarRS != -CAR_SPIN_SPEED) {
      lastCarLS = CAR_SPIN_SPEED; lastCarRS = -CAR_SPIN_SPEED;
      tbDrive(lastCarLS, lastCarRS);
    }
    return;
  }
  int16_t lS = stickToSpeed(ps2.ly());
  int16_t rS = stickToSpeed(ps2.ry());
  if (abs(lS) < CAR_SPEED_DEADZONE) lS = 0;
  if (abs(rS) < CAR_SPEED_DEADZONE) rS = 0;
  lS = constrain(lS, -CAR_MAX_SPEED, CAR_MAX_SPEED);
  rS = constrain(rS, -CAR_MAX_SPEED, CAR_MAX_SPEED);
  if (lS != lastCarLS || rS != lastCarRS) {
    lastCarLS = lS; lastCarRS = rS;
    tbDrive(lS, rS);
  }
}

// ---------- 串口命令（$G2 + 车测试 + 舵机测试 + 巡线） ----------
void lnPrintStatus() {
  Serial.print("LINE 状态=");
  switch (lnState) {
    case LS_FOLLOW:  Serial.print("FOLLOW"); break;
    case LS_PARKING: Serial.print("PARKING"); break;
    case LS_CLEAR:   Serial.print("CLEAR"); break;
    case LS_IDLE:    Serial.print("IDLE"); break;
  }
  Serial.print(" 速度="); Serial.print(lnBaseSpeed);
  Serial.print(" Kp="); Serial.print(lnPid.kp);
  Serial.print(" Ki="); Serial.print(lnPid.ki);
  Serial.print(" Kd="); Serial.println(lnPid.kd);
}

void command(const char *line) {
  if (!strcmp(line, "start")) sendCommand("START");
  else if (!strcmp(line, "track")) sendCommand("TRACK");
  else if (!strcmp(line, "stop") || !strcmp(line, "hold")) {
    if (gimbalMode == MODE_LINE) { lnState = LS_IDLE; tbDrive(0, 0); Serial.println("LINE: 手动停车 (go 恢复)"); }
    else sendCommand("HOLD");
  }
  else if (!strcmp(line, "off")) sendCommand("OFF");
  else if (!strcmp(line, "home")) {          // 本机回中(舵机归本机驱动), 再通知 OpenMV
    panUs = tiltUs = SERVO_MID_US;
    setServoUs(PIN_PAN, panUs); setServoUs(PIN_TILT, tiltUs);
    aimSyncFromServo();
    sendCommand("HOME");
  }
  else if (!strcmp(line, "calpan")) sendCommand("CALPAN");
  else if (!strcmp(line, "ping") || !strcmp(line, "p")) sendCommand("PING");
  else if (!strcmp(line, "status")) sendCommand("STATUS");
  else if (!strcmp(line, "info")) sendCommand("INFO");
  else if (!strcmp(line, "limits")) sendCommand("LIMITS");
  else if (!strcmp(line, "pan+")) aimJog(0, 20);
  else if (!strcmp(line, "pan-")) aimJog(0, -20);
  else if (!strcmp(line, "tilt+")) aimJog(1, 20);
  else if (!strcmp(line, "tilt-")) aimJog(1, -20);
  else if (!strcmp(line, "accept")) sendCommand("ACCEPT");
  // ---- 视觉跟踪 ----
  else if (!strcmp(line, "vision") || !strcmp(line, "vt")) switchMode(MODE_TRACK);
  else if (!strcmp(line, "aim")) aimPrint();
  else if (!strcmp(line, "tgt")) {
    Serial.print("TGT last=$G2,TGT,"); Serial.print(aimCx); Serial.print(",");
    Serial.print(aimCy); Serial.print(","); Serial.print(aimLx); Serial.print(",");
    Serial.print(aimLy); Serial.print(","); Serial.print(aimMode); Serial.print(",");
    Serial.print(aimRun ? 1 : 0); Serial.print("! frames="); Serial.print(aimFrames);
    Serial.print(" bad="); Serial.println(aimBadFrames);
  }
  else if (!strcmp(line, "cf")) { tbDrive(200, 200); delay(1000); tbDrive(0, 0); lastCarLS = lastCarRS = 0; }
  else if (!strcmp(line, "cb")) { tbDrive(-200, -200); delay(1000); tbDrive(0, 0); lastCarLS = lastCarRS = 0; }
  else if (!strcmp(line, "cs")) { tbDrive(0, 0); lastCarLS = lastCarRS = 0; }
  else if (!strncmp(line, "ch ", 3)) {
    int side = 0, spd = 0, t = 0;
    if (sscanf(line + 3, "%d %d %d", &side, &spd, &t) == 3 &&
        (side == 0 || side == 1) && spd >= -255 && spd <= 255 && t >= 0 && t <= 9999) {
      tbDrive(side == 0 ? spd : 0, side == 1 ? spd : 0);
      if (t > 0) { delay(t); tbDrive(0, 0); }
      lastCarLS = lastCarRS = 0;
    } else Serial.println("usage: ch <side 0=left 1=right> <speed -255..255> <time ms>");
  }
  else if (!strncmp(line, "s1 ", 3)) {
    int u = 0;
    if (sscanf(line + 3, "%d", &u) == 1 && u >= 0 && u <= 3000) {
      panUs = u; setServoUs(PIN_PAN, panUs); aimSyncFromServo();
    } else Serial.println("usage: s1 <us 500..2500>");
  }
  else if (!strncmp(line, "s2 ", 3)) {
    int u = 0;
    if (sscanf(line + 3, "%d", &u) == 1 && u >= 0 && u <= 3000) {
      tiltUs = u; setServoUs(PIN_TILT, tiltUs); aimSyncFromServo();
    } else Serial.println("usage: s2 <us 500..2500>");
  }
  else if (!strcmp(line, "sh")) {
    panUs = tiltUs = SERVO_MID_US;
    setServoUs(PIN_PAN, panUs); setServoUs(PIN_TILT, tiltUs);
    aimSyncFromServo();
  }
  // ---- 巡线命令 ----
  else if (!strcmp(line, "line")) switchMode(MODE_LINE);
  else if (!strcmp(line, "manual")) switchMode(MODE_CAR);
  else if (!strcmp(line, "1")) { lnBaseSpeed = SPEED_LOW;  Serial.print("速度=低速 "); Serial.println(lnBaseSpeed); }
  else if (!strcmp(line, "2")) { lnBaseSpeed = SPEED_MED;  Serial.print("速度=中速 "); Serial.println(lnBaseSpeed); }
  else if (!strcmp(line, "3")) { lnBaseSpeed = SPEED_HIGH; Serial.print("速度=高速 "); Serial.println(lnBaseSpeed); }
  else if (!strncmp(line, "kp ", 3)) { lnPid.kp = atof(line + 3); Serial.print("Kp="); Serial.println(lnPid.kp); }
  else if (!strncmp(line, "ki ", 3)) { lnPid.ki = atof(line + 3); Serial.print("Ki="); Serial.println(lnPid.ki); }
  else if (!strncmp(line, "kd ", 3)) { lnPid.kd = atof(line + 3); Serial.print("Kd="); Serial.println(lnPid.kd); }
  else if (!strcmp(line, "go")) {
    if (gimbalMode == MODE_LINE) { lnState = LS_FOLLOW; lnResetPID(); Serial.println("LINE: 恢复循迹"); }
    else Serial.println("当前不是巡线模式 (输入 line 进入)");
  }
  else if (!strcmp(line, "park")) {
    if (gimbalMode == MODE_LINE && lnState == LS_FOLLOW) lnEnterParking();
    else Serial.println("仅巡线模式可用");
  }
  else if (!strcmp(line, "s")) lnPrintStatus();
  else if (!strcmp(line, "pollon")) { polling = true; lastPoll = millis(); Serial.println("POLL ON"); }
  else if (!strcmp(line, "polloff")) { polling = false; Serial.println("POLL OFF"); }
  else if (!strcmp(line, "stats")) {
    Serial.print("STATS req="); Serial.print(requests);
    Serial.print(" rep="); Serial.print(replies);
    Serial.print(" timeouts="); Serial.print(timeouts);
    Serial.print(" laser_hits="); Serial.print(laserHitCount);
    Serial.print(" tgt_frames="); Serial.print(aimFrames);
    Serial.print(" tgt_bad="); Serial.println(aimBadFrames);
  } else if (*line) help();
}

void help() {
  Serial.println("V1-13 console (TB6612 + PS2 + servo PWM + line follow + vision aim); newline commands:");
  Serial.println("$G2: start|track|stop/hold|off|home|calpan|pan+/pan-/tilt+/tilt-|ping|status|info|limits|stats|pollon|polloff");
  Serial.println("Car: cf/cb/cs, ch <side 0/1> <speed> <ms>");
  Serial.println("Servo: s1 <us> = pan, s2 <us> = tilt, sh = 回中");
  Serial.println("Line: line 进入巡线, manual 退出, 1/2/3 速度, kp/ki/kd <值>, stop=手动停车, go=恢复, park=测试停车, s=状态");
  Serial.println("Vision: vision 进入 TRACK(消费 OpenMV $G2,TGT), aim 看瞄准状态, tgt 看最近视觉帧");
  Serial.println("PS2: SELECT 循环 CAR/GIMBAL/LINE/TRACK; CAR: 摇杆=电机, L3/R3=旋转; GIMBAL: L-X=pan R-Y=tilt, START=回中");
}

// ---------- UART2 接收: $G2 帧 + DONE 裸串 ----------
void handleBus() {
  static char lineBuf[24];
  static uint8_t lineBufLen = 0;
  while (Bus.available()) {
    char c = static_cast<char>(Bus.read());
    if (c == '$') { lineBufLen = 0; receiveByte(c); continue; }
    if (c == '\n' || c == '\r') {
      if (lineBufLen > 0) {
        lineBuf[lineBufLen] = '\0';
        if (!strcmp(lineBuf, CMD_DONE) && lnState == LS_PARKING) lnResumeFromParking();
        lineBufLen = 0;
      }
    } else if (lineBufLen < sizeof(lineBuf) - 1) {
      lineBuf[lineBufLen++] = c;
    }
    receiveByte(c);   // 非 $ 字符在 $G2 解析里会被忽略
  }
}

// ---------- setup / loop ----------
void setup() {
  Serial.begin(BAUD);
  Bus.begin(BAUD, SERIAL_8N1, BUS_RX, BUS_TX);

  // 电机方向脚（STBY 硬件接 3V3 常通）
  pinMode(PIN_AIN1_L, OUTPUT); pinMode(PIN_AIN2_L, OUTPUT);
  pinMode(PIN_AIN1_R, OUTPUT); pinMode(PIN_AIN2_R, OUTPUT);
  ledcAttach(PIN_PWM_L, TB_PWM_FREQ, TB_PWM_RES);
  ledcAttach(PIN_PWM_R, TB_PWM_FREQ, TB_PWM_RES);

  // 云台舵机 50Hz 16bit, 回中
  ledcAttach(PIN_PAN, SERVO_FREQ, SERVO_RES);
  ledcAttach(PIN_TILT, SERVO_FREQ, SERVO_RES);
  setServoUs(PIN_PAN, panUs);
  setServoUs(PIN_TILT, tiltUs);
  aimSyncFromServo();

  // 8 路循迹传感器 I2C
  Wire.begin(I2C_SDA, I2C_SCL, I2C_FREQ);

  ps2.begin();
  carStop();
  lastPoll = lastPs2Poll = lastValidPs2 = millis();

  Serial.println("V1-13 READY: TB6612 + PS2 + servo PWM + line follow + vision aim");
  Serial.println("TB6612 左: PWM32/AIN1_13/AIN2_12  右: PWM33/AIN1_2/AIN2_4  STBY=3V3常通");
  Serial.println("舵机: PAN=27 TILT=14 (PWM接口4/5)  循迹: I2C 0x5D SDA21/SCL22");
  Serial.print("PS2 pins: DAT="); Serial.print(PS2_DAT);
  Serial.print(" CMD="); Serial.print(PS2_CMD);
  Serial.print(" ATT="); Serial.print(PS2_ATT);
  Serial.print(" CLK="); Serial.println(PS2_CLK);
  help();

  // 上电默认模式 (BOOT_MODE, 见模式段). 进 TRACK 会立刻发一次 $G2,TRACK;
  // 万一 OpenMV 还没起来没收到, V1-11 的自愈会每 2s 重发直到它应答.
  switchMode(BOOT_MODE);
}

void loop() {
  uint32_t now = millis();
  if (rxUsed && now - lastRx > 100) rxUsed = 0;
  if (pending && now - sentAt >= TIMEOUT_MS) {
    pending = false; ++timeouts;
    Serial.println("TIMEOUT: execution state unknown.");
  }

  // ---- PS2 轮询(20ms) ----
  static uint8_t ps2Streak = 0;
  if (now - lastPs2Poll >= PS2_POLL_MS) {
    lastPs2Poll = now;
    ps2.writeRead();
    if (ps2.dataReady()) { if (ps2Streak < 250) ++ps2Streak; lastValidPs2 = now; }
    else ps2Streak = 0;
  }

  // ---- 调试打印: 每 500ms 手柄帧; 巡线模式另打巡线状态 ----
  static uint32_t lastDbg = 0;
  if (now - lastDbg >= 500) {
    lastDbg = now;
    if (gimbalMode == MODE_LINE) {
      if (now - lnLastDebugMs >= LINE_DEBUG_MS) {
        lnLastDebugMs = now;
        char bits[9];
        for (int i = 0; i < 8; i++) bits[i] = (lnSensorBits & (1 << i)) ? '1' : '0';
        bits[8] = '\0';
        char pbuf[40];
        snprintf(pbuf, sizeof(pbuf), " pos=%.2f err=%.2f", lnPosition, lnError);
        Serial.print("[LINE] S="); Serial.print(bits);
        Serial.print(" 黑="); Serial.print(lnBlackCount);
        Serial.print(pbuf);
        Serial.print(" L="); Serial.print(lnLSpeed);
        Serial.print(" R="); Serial.print(lnRSpeed);
        Serial.print(" v="); Serial.print(lnBaseSpeed);
        Serial.println(lnSensorOk ? " i2c=ok" : " i2c=FAIL");
      }
    } else if (gimbalMode == MODE_TRACK) {
      Serial.print("[AIM] "); aimPrint();
    } else {
      char tmp[32], mhex[4];
      snprintf(tmp, sizeof(tmp), "%02X %02X %02X %02X %02X %02X %02X %02X %02X",
               ps2.byte(0), ps2.byte(1), ps2.byte(2), ps2.byte(3), ps2.byte(4),
               ps2.byte(5), ps2.byte(6), ps2.byte(7), ps2.byte(8));
      snprintf(mhex, sizeof(mhex), "%02X", ps2.modeByte());
      Serial.print("PS2["); Serial.print(tmp);
      Serial.print("] mode="); Serial.print(mhex);
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
  }

  const bool handleOk = ps2Streak >= 3;
  if (handleOk) {
    // SELECT 按下沿 -> 三态循环: CAR -> GIMBAL -> LINE -> CAR
    bool selPressed = ps2.pressed(PS2_SELECT);
    if (selPressed && !selWasPressed) {
      switchMode((gimbalMode + 1) % 4);
    }
    selWasPressed = selPressed;

    bool startPressed = ps2.pressed(PS2_START);
    if (startPressed && !startWasPressed) {
      if (gimbalMode == MODE_GIMBAL) {
        panUs = tiltUs = SERVO_MID_US;
        setServoUs(PIN_PAN, panUs); setServoUs(PIN_TILT, tiltUs);
        aimSyncFromServo();
      } else if (gimbalMode == MODE_LINE) {
        lnState = LS_IDLE; tbDrive(0, 0);   // 巡线手动停车
        Serial.println("LINE: 手动停车 (go 恢复)");
      } else if (gimbalMode == MODE_TRACK) {
        aimRun = false;                     // 冻结云台, 并让 OpenMV 停发瞄准数据
        aimVelUs[0] = aimVelUs[1] = 0.0f;
        aimReason = "HOLD";
        sendCommand("HOLD");
        Serial.println("TRACK: 暂停瞄准 (已发 $G2,HOLD)");
      } else {
        carStop();
        lastCarLS = lastCarRS = 0;
      }
    }
    startWasPressed = startPressed;

    if (gimbalMode == MODE_CAR) handleCar();
    else if (gimbalMode == MODE_GIMBAL) handleGimbal();
  } else {
    startWasPressed = true;
    selWasPressed = true;
    if (gimbalMode == MODE_CAR && (lastCarLS != 0 || lastCarRS != 0)) {
      carStop();
      lastCarLS = lastCarRS = 0;
    }
  }

  // ---- 巡线控制(10ms) ----
  if (gimbalMode == MODE_LINE && now - lnLastControlMs >= LINE_CTRL_MS) {
    lnLastControlMs = now;
    lnRunControl((float)LINE_CTRL_MS / 1000.0f);
  }

  // ---- 视觉瞄准控制(20ms, 仅 TRACK 模式) ----
  if (gimbalMode == MODE_TRACK) {
    // 自愈: OpenMV 若正在重启(停止脚本后看门狗会在 ~8s 后复位一次并重跑 main.py),
    // 那一刻发出的 $G2,TRACK 会丢, OpenMV 就停在 run=0, 云台永远不动.
    // 所以只要还没收到 run=1, 就每 2 秒重发一次 TRACK, 直到它应答为止.
    if (!aimRun && !pending && now - aimLastTrackCmd >= 2000) {
      aimLastTrackCmd = now;
      Serial.println("TRACK: no run=1 yet -> re-asserting $G2,TRACK");
      sendCommand("TRACK");
    }
    aimControlUpdate(now);
  }

  // ---- UART2 接收: $G2 + DONE ----
  handleBus();

  // ---- USB 串口命令 ----
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

  // ---- $G2 主动轮询(默认关) ----
  if (polling && !pending && millis() - lastPoll >= POLL_MS) {
    sendCommand(nextInfo ? "INFO" : "STATUS", nullptr, true);
    nextInfo = !nextInfo;
  }
  delay(2);
}
