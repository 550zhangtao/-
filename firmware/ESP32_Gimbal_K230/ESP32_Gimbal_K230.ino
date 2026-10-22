/*
 * ESP32_Gimbal_K230 —— K230 视觉云台主控（精简版）
 * =================================================
 * 只保留两件事：
 *   1. 与 K230 的 UART 通信（UART2，115200）
 *   2. 双舵机云台控制（PAN=27 / TILT=14，50Hz PWM）
 *
 * 已删除：PS2 手柄、TB6612 电机、巡线 I2C、模式切换、急停电机等全部无关逻辑。
 *
 * 接线：
 *   K230 GPIO3 (UART1_TX)  -> ESP32 GPIO16 (UART2_RX)
 *   K230 GPIO4 (UART1_RX)  <- ESP32 GPIO17 (UART2_TX)
 *   GND 共地，舵机用独立 5V 电源（不要从 ESP32 3.3V 取舵机电流）
 *
 * K230 -> ESP32 协议（每帧一条，\n 结尾）：
 *   $KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!
 *     seq   递增序号
 *     run   1=有目标可跟踪，0=无目标
 *     cx,cy 目标框中心（640x480 像素）
 *     bx,by 靶心中心；无靶心时 -1
 *     conf  置信度 0..1000
 *     （共 7 个数字字段）
 *
 * ESP32 -> K230 应答：
 *   $E,ACK,<seq>,<state>,<pan_us>,<tilt_us>!
 *     state 0=空闲 1=跟踪 2=已对准 3=数据超时
 *
 * USB 串口（115200）调试命令：
 *   s1 <us> / s2 <us>   直接设置 pan/tilt（500..2500）
 *   pan <±us> / tilt <±us>  相对微调
 *   sh                  回中
 *   auto / manual       自动跟踪 / 手动
 *   sweep               舵机扫动测试
 *   status              状态
 *   $KT,...             模拟一帧 K230 数据，用于无 K230 时测试
 *
 * 2026-10-20 补齐（模块取自 ESP32_Finished_V19_LineMotor，上面云台部分未改）：
 *   - PS2 手柄（L1/L2 小车、R1/R2 云台、X 急停、△ 回中、START、方向键）
 *   - TB6612 双电机（限幅/斜坡/换向静置/急停锁存/STBY 待机、USB 诊断命令）
 *   - 8 路 I2C 循迹（PID + 停车标记 + 丢线保护 + 停车/继续握手）
 *   - 两套并行模式 carMode / gimbalMode（小车上电 NONE，云台**上电即 TRACK**）
 *   - 云台跟踪只依赖 $KT，不依赖手柄：手柄掉线只断电机，不再关跟踪
 *   新增命令见 help() 末尾的 helpExtra() 段落。
 */

#include <Arduino.h>
#include "soc/ledc_struct.h"
#include <Wire.h>
#include <math.h>
#include <string.h>

// ---------------- 舵机 ----------------
constexpr uint8_t PIN_PAN = 27;
constexpr uint8_t PIN_TILT = 14;
constexpr int SERVO_FREQ = 50;
constexpr int SERVO_RES = 16;
constexpr int SERVO_MIN_US = 500;
constexpr int SERVO_MAX_US = 2500;
constexpr int SERVO_MID_US = 1500;
// LEDC 通道（group0 的 0/1）：直接写寄存器需要已知通道号
constexpr uint8_t CH_PAN = 0;
constexpr uint8_t CH_TILT = 1;


// 机械行程（沿用原 V19 的安全范围）
constexpr int PAN_MIN_US = 740;
constexpr int PAN_MAX_US = 2260;
constexpr int TILT_MIN_US = 1200;
constexpr int TILT_MAX_US = 1800;

// 方向：如果云台追反了，把对应 SIGN 改成 -1
constexpr int8_t PAN_SIGN = 1;
constexpr int8_t TILT_SIGN = 1;

// ---------------- 视觉跟踪参数 ----------------
constexpr int FRAME_W = 640;
constexpr int FRAME_H = 480;
constexpr int CENTER_X = FRAME_W / 2;
constexpr int CENTER_Y = FRAME_H / 2;

constexpr int AIM_DEADBAND_X = 8;      // 像素死区（越小越跟手，太小会抖）
constexpr int AIM_DEADBAND_Y = 8;
// PID 默认值：可用 USB 命令在线改，无需重烧（pid kp ki kd / pid2 kp ki kd）
constexpr float PAN_KP_DEF = 2.5f, PAN_KI_DEF = 0.3f, PAN_KD_DEF = 0.15f;
constexpr float TILT_KP_DEF = 2.5f, TILT_KI_DEF = 0.2f, TILT_KD_DEF = 0.10f;
constexpr float PID_D_FILTER = 0.35f;  // 微分项低通，抑制检测噪声
constexpr float PAN_MAX_RATE = 2600.0f; // us/s（≈234°/s，接近舵机空载极限）
constexpr float TILT_MAX_RATE = 1000.0f;
constexpr float PAN_ACCEL = 40000.0f;  // us/s^2
constexpr float TILT_ACCEL = 20000.0f;
// 目标预处理 + 速度自适应前视
constexpr float WBAND_PX = 8.0f;      // 死区（像素）
constexpr int TGT_JUMP_PX = 80;       // 单帧跳变超过该值先当离群点
constexpr uint8_t TGT_JUMP_HOLD = 4;  // 连续 N 帧都跳才认定目标真的换了位置
float tgtAlpha = 0.35f;               // 目标位置低通（越大越跟手、越抖）
float trkLead  = 0.20f;               // 最大前视时间 s（随目标速度自适应）
float velTau   = 0.02f;               // 速度指令一阶低通时间常数 s（越大越丝滑、越滞后）
constexpr uint32_t AIM_CTRL_MS = 10;   // 100Hz 控制环
constexpr uint32_t TARGET_TIMEOUT_MS = 400;

// ---------------- UART ----------------
constexpr int BUS_RX = 16;
constexpr int BUS_TX = 17;
constexpr uint32_t BAUD = 115200;

HardwareSerial K230(2);

// ============================================================================
// 【新增】以下模块从 ESP32_Finished_V19_LineMotor 补齐（本文件原有云台/通信部分未改）
//   - PS2 手柄接收（DAT/CMD/ATT/CLK）
//   - TB6612 双电机驱动（限幅 / 斜坡 / 换向静置 / 急停锁存 / STBY 待机）
//   - 8 路 I2C 循迹（PID + 停车标记 + 丢线保护）
//   - 两套并行模式：carMode（小车）/ gimbalMode（云台）
// 舵机写入统一走上面的 applyServos()（按 LEDC 通道直写寄存器），不再用 ledcWrite。
// ============================================================================

// ---- 新增：结构体要先于"第一个函数定义"，Arduino 自动原型就插在那之前 ----
struct PID {
  float kp, ki, kd;
  float integral, prevError, integralMax;
  bool  hasPrev;
  float derivative;
};
struct MotorSide {
  uint8_t  ain1, ain2, pwmPin;
  int16_t  target;      // 期望占空比 -MAX..MAX
  int16_t  applied;     // 当前真正输出的占空比
  int8_t   dir;         // 当前方向脚状态 1/-1/0
  int8_t   lastDir;     // 最近一次非零转动方向
  bool     reversing;   // 正在等换向静置窗口
  uint32_t zeroAt;      // 输出刚降到 0 的时刻
  bool     brake;       // 正在短路刹车
  uint32_t brakeStart;
};
struct BtnEdge {
  uint16_t mask;
  bool     was;
};

// ---- 新增：PS2 手柄接收器引脚（DAT/CMD/ATT/CLK 对应接收器上 DI/DO/CS/CK）----
constexpr uint8_t PS2_DAT = 19, PS2_CMD = 18, PS2_ATT = 15, PS2_CLK = 23;

// 按键位（bit=0 表示按下, active-low）
enum : uint16_t {
  PS2_L2 = 1 << 0, PS2_R2 = 1 << 1, PS2_L1 = 1 << 2, PS2_R1 = 1 << 3,
  PS2_TRI = 1 << 4, PS2_CIRC = 1 << 5, PS2_CROSS = 1 << 6, PS2_SQUARE = 1 << 7,
  PS2_SELECT = 1 << 8, PS2_L3 = 1 << 9, PS2_R3 = 1 << 10, PS2_START = 1 << 11,
  PS2_UP = 1 << 12, PS2_RIGHT = 1 << 13, PS2_DOWN = 1 << 14, PS2_LEFT = 1 << 15
};

// ---------- PS2 位操作驱动（V19 原样移植; 移植自 Carbot 出厂代码 z_ps2.py） ----------
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
  bool writeRead() {
    uint8_t candidate[9];
    digitalWrite(att_, LOW);
    candidate[0] = transfer(0x01); candidate[1] = transfer(0x42);
    for (int i = 2; i < 9; ++i) candidate[i] = transfer(candidate[0]);
    digitalWrite(att_, HIGH);
    if (candidate[0] != 0xFF || candidate[1] != 0x73 || candidate[2] != 0x5A) return false;
    memcpy(buf_, candidate, sizeof(buf_)); // Never replace good axes/buttons with a bad frame.
    return true;
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
      // 10us 半周期(50kHz): V19 的 5us(100kHz) 已接近 PS2 上限, 克隆接收器常读不出来
      digitalWrite(clk_, HIGH); delayMicroseconds(10);
      digitalWrite(clk_, LOW); delayMicroseconds(10);
      if (digitalRead(dat_) == HIGH) readData |= 0x01 << i;
    }
    digitalWrite(clk_, HIGH); delayMicroseconds(10);
    return readData;
  }
  uint8_t dat_, cmd_, att_, clk_;
  uint8_t buf_[9];
};

// ---- 新增：小车 TB6612 驱动（AIN 用普通 GPIO, 不占 PWM 舵机接口; STBY 用 GPIO25）----
constexpr uint8_t PIN_PWM_L = 33;    // left physical motor: PWM1 signal
constexpr uint8_t PIN_AIN1_L = 2;    // left: S3 SD3
constexpr uint8_t PIN_AIN2_L = 4;    // left: S3 SA3
constexpr uint8_t PIN_PWM_R = 32;    // right physical motor: PWM0 signal
constexpr uint8_t PIN_AIN1_R = 13;   // right: S4 SD4
constexpr uint8_t PIN_AIN2_R = 26;   // right: PWM3 (原 GPIO12 坏, 已改 26)
constexpr uint8_t PIN_STBY = 25;     // 02 board PWM2 signal; HIGH=enable, LOW=standby
static_assert(PIN_STBY <= 33, "STBY requires an output-capable GPIO; GPIO34-39 are input-only.");
static_assert(PIN_STBY != PIN_PWM_L && PIN_STBY != PIN_PWM_R &&
              PIN_STBY != PIN_AIN1_L && PIN_STBY != PIN_AIN2_L &&
              PIN_STBY != PIN_AIN1_R && PIN_STBY != PIN_AIN2_R,
              "STBY conflicts with a motor control pin.");
constexpr int    TB_PWM_FREQ = 1000;   // PWM 频率 Hz（电机走的通道, 与舵机 50Hz 不同定时器）
constexpr int    TB_PWM_RES = 8;       // 8bit -> 0~255
// 电机占 LEDC 通道 2/3；舵机在上面固定占 0/1，两者分属不同定时器，互不影响
constexpr uint8_t CH_MOTOR_L = 2;
constexpr uint8_t CH_MOTOR_R = 3;
constexpr bool   TB_INV_L = false;     // 左方向反了改 true
constexpr bool   TB_INV_R = false;     // 右方向反了改 true
constexpr int16_t CAR_MAX_SPEED = 120;
constexpr int16_t CAR_SPEED_DEADZONE = 30;
constexpr int16_t CAR_SPIN_SPEED = 90;
constexpr uint32_t PS2_POLL_MS = 20;

// Motor software bounds: no current feedback; compatibility remains a hardware check.
int16_t  MOTOR_MAX_DUTY = 120;                    // software duty cap, not current limiting
constexpr uint32_t MOTOR_CTRL_MS = 10;            // 电机斜坡控制周期
constexpr float    MOTOR_SLEW_PER_S = 300.0f;     // PWM/秒; 越小越温柔也越迟钝
constexpr uint32_t MOTOR_REVERSE_DEAD_MS = 500;   // 换向静置窗口（不是实测转子停转）
constexpr uint32_t MOTOR_BRAKE_HOLD_MS = 0;       // 短路刹车关闭
constexpr int16_t  MOTOR_HARD_CAP = 120;
constexpr uint32_t MOTOR_TEST_START_TIMEOUT_MS = 1500;

// ---- 新增：云台手动微调（V19 原值）----
constexpr int16_t  GIMBAL_DEADZONE = 20;
constexpr int16_t  GIMBAL_STEP_DIV = 4;    // 步进: 满偏≈310us/s (想更慢改 6/8)
constexpr uint32_t GIMBAL_POLL_MS = 100;

// ---- 新增：8 路循迹传感器（I2C 模式, S6 口 SDA21/SCL22, 5V 供电）----
constexpr uint8_t  I2C_SDA = 21, I2C_SCL = 22;
constexpr uint8_t  I2C_ADDR = 0x5D;
constexpr uint8_t  I2C_REG_STATE = 5;
constexpr uint32_t I2C_FREQ = 400000;
constexpr bool     SENSOR_FLIP = false;    // 通道1..8 与左→右相反时改 true

// ---- 新增：巡线/停车参数（V19 原值, 移植自 ESP32_LineCar.ino）----
constexpr int     PARKING_MIN_BLACK = 6;         // 同时压黑 >= 该路数判定停车标记
constexpr uint32_t PARKING_DEBOUNCE_MS = 80;
constexpr float   KP_DEFAULT = 18.0f, KI_DEFAULT = 0.0f, KD_DEFAULT = 0.08f;
constexpr int     SPEED_LOW = 70, SPEED_MED = 90, SPEED_HIGH = 110;
constexpr uint32_t CLEAR_MARKER_MIN_MS = 150;
constexpr uint32_t CLEAR_MARKER_MAX_MS = 2000;
constexpr uint32_t PARK_BLANK_MS = 300;
constexpr uint32_t LOST_TIMEOUT_MS = 250;
constexpr float   LOST_SPEED_FACTOR = 0.6f;
constexpr uint32_t LINE_CTRL_MS = 10;
constexpr uint32_t LINE_DEBUG_MS = 200;
const char CMD_DONE[] = "DONE";       // K230 回 DONE = 打靶完成, 巡线继续
const char RSP_PARKED[] = "PARKED";   // 停车停稳后发给 K230

// ---- 新增：手柄就绪判据（V19 原样, 故意 constexpr 以便 static_assert）----
constexpr uint32_t PAD_NEUTRAL_HOLD_MS = 500;
constexpr uint32_t PAD_FRESH_MS = 100;
constexpr bool padKeepReady(bool wasReady, uint8_t streak, uint32_t age);
constexpr bool padKeepReady(bool wasReady, uint8_t streak, uint32_t age) {
  return streak >= 3 || (wasReady && age <= PAD_FRESH_MS);
}
static_assert(!padKeepReady(false, 2, 0), "Need initial three good frames");
static_assert(padKeepReady(true, 0, 20), "One short pad gap keeps cached good input");
static_assert(!padKeepReady(true, 0, 101), "Cached pad input expires after 100ms");

// 电机使能 / 换向静置判据（V19 原样, constexpr + static_assert 自检）
constexpr bool motorEnableAllowed(bool ready, bool armed, int16_t lt, int16_t rt,
                                  int16_t la, int16_t ra);
constexpr bool motorEnableAllowed(bool ready, bool armed, int16_t lt, int16_t rt,
                                  int16_t la, int16_t ra) {
  return ready && armed && (lt != 0 || rt != 0 || la != 0 || ra != 0);
}
constexpr bool motorReverseWait(int8_t want, int8_t last, int16_t applied,
                                uint32_t now, uint32_t zeroAt, uint32_t dwell);
constexpr bool motorReverseWait(int8_t want, int8_t last, int16_t applied,
                                uint32_t now, uint32_t zeroAt, uint32_t dwell) {
  return want != 0 && last != 0 && want != last &&
         (applied != 0 || uint32_t(now - zeroAt) < dwell);
}
static_assert(!motorEnableAllowed(false, true, 100, 0, 0, 0), "PWM failure must disable");
static_assert(!motorEnableAllowed(true, false, 100, 0, 0, 0), "Unarmed target must disable");
static_assert(!motorEnableAllowed(true, true, 0, 0, 0, 0), "Idle must disable");
static_assert(motorEnableAllowed(true, true, 0, 0, -20, 0), "Ramp-down must remain enabled");
static_assert(motorEnableAllowed(true, true, 0, -50, 0, 0), "Reverse target must enable");
static_assert(motorReverseWait(-1, 1, 20, 900, 100, 500), "Moving reversal must wait");
static_assert(motorReverseWait(-1, 1, 0, 599, 100, 500), "499ms is not enough");
static_assert(!motorReverseWait(-1, 1, 0, 600, 100, 500), "500ms releases reversal");
static_assert(!motorReverseWait(1, 1, 20, 110, 100, 500), "Same direction must work");
static_assert(!motorReverseWait(-1, 1, 0, 300, 0xffffff00u, 500), "Timer wrap must work");

// ---------------- 状态 ----------------
int panUs = SERVO_MID_US;
int tiltUs = SERVO_MID_US;
float panVel = 0.0f;
float tiltVel = 0.0f;
float panKp = PAN_KP_DEF, panKi = PAN_KI_DEF, panKd = PAN_KD_DEF;
float tiltKp = TILT_KP_DEF, tiltKi = TILT_KI_DEF, tiltKd = TILT_KD_DEF;
float panInteg = 0.0f, tiltInteg = 0.0f;
float panPrevE = 0.0f, tiltPrevE = 0.0f, panDFilt = 0.0f, tiltDFilt = 0.0f;
bool pidInit = false;
bool autoTrack = true;

int lastCx = -1, lastCy = -1;
int lastBx = -1, lastBy = -1;
int lastConf = 0;
bool lastRun = false;
uint32_t lastFrameMs = 0;
uint32_t frames = 0, badFrames = 0;
uint32_t lastAckSeq = 0;
int trackState = 0;  // 0 idle, 1 tracking, 2 aligned, 3 stale
char lastLine[128] = "-";

// ---- 靶心丢失 -> 自动复位 ----
// 靶心连续 N 帧未检出或 900ms 无 bull -> 判定打靶完成，云台回 NONE（不改 carMode）。
// 实测 N：现场 60 帧最大连续漏检 0；新视角 20 帧最大 2 -> N=10 + 900ms 兜底。
constexpr uint32_t AIM_LOST_FRAMES = 10;
constexpr uint32_t AIM_LOST_MS = 900;
constexpr uint32_t AIM_START_GRACE_MS = 3000;   // K230 启动约 1.5s，留余量
constexpr int AIM_BULL_MIN_CONF = 300;
uint32_t aimBullLostFrames = 0;
uint32_t aimLastBullMs = 0;
bool aimEverBull = false;
uint32_t aimStartMs = 0;
float aimErrX = 0.0f, aimErrY = 0.0f;   // 当前控制误差（靶心 - 画面中心）

// ---- 新增：手柄 / 模式 / 电机 / 循迹 运行状态 ----
PS2Controller ps2(PS2_DAT, PS2_CMD, PS2_ATT, PS2_CLK);
uint32_t lastPs2Poll = 0, lastValidPs2 = 0, padBadFrames = 0, padNeutralSince = 0;
bool padIsValid = false, padLostStopped = false;
// 从未见过手柄时（只用手柄做可选外设/纯云台使用）不启用"失联保护",
// 否则没接手柄会把 USB 打开的跟踪/电机误判成掉线而关掉。
bool padEverSeen = false;

// 按键边沿检测: was 初值 true -> 上电时正按着的键不会立刻误触发, 必须先松开再按.
BtnEdge btnCar    = {PS2_L1, true};
BtnEdge btnGimbal = {PS2_R1, true};
BtnEdge btnLine   = {PS2_L2, true};
BtnEdge btnTrack  = {PS2_R2, true};
BtnEdge btnSel    = {PS2_SELECT, true};
BtnEdge btnStart  = {PS2_START, true};
BtnEdge btnCross  = {PS2_CROSS, true};
BtnEdge btnTri    = {PS2_TRI, true};
BtnEdge btnSq     = {PS2_SQUARE, true};
BtnEdge btnUp     = {PS2_UP, true};
BtnEdge btnDown   = {PS2_DOWN, true};

// 两套互相独立的状态: L1/L2 管小车, R1/R2 管云台, 可并行
//   carMode   : CAR_NONE(上电默认, 电机不使能) / CAR_MANUAL(手柄开) / CAR_LINE(循迹)
//   gimbalMode: GIMBAL_NONE(上电默认, 保持不动) / GIMBAL_MANUAL(手柄微调) / GIMBAL_TRACK($KT)
enum : uint8_t { CAR_NONE = 0, CAR_MANUAL = 1, CAR_LINE = 2 };
enum : uint8_t { GIMBAL_NONE = 0, GIMBAL_MANUAL = 1, GIMBAL_TRACK = 2 };
constexpr uint8_t BOOT_CAR_MODE = CAR_NONE;       // 上电小车不使能, 需手柄按 L1/L2 选
// 上电默认无模式: 云台保持不动, 也不会让 K230 跑识别;
// 必须按手柄 MODE(SELECT) 再按 R2 才进跟踪、K230 才开始跑。
constexpr uint8_t BOOT_GIMBAL_MODE = GIMBAL_NONE;
uint8_t carMode = BOOT_CAR_MODE;
uint8_t gimbalMode = BOOT_GIMBAL_MODE;
uint32_t lastGimbalSend = 0;

// ---- 新增：电机状态 ----
bool carEStop = true;
bool motorArmed = false, motorPwmReady = false, motorArmFromPad = false;
bool motorTestActive = false, motorTestOutputStarted = false, motorStandbyCommand = false;
uint32_t motorTestStart = 0, motorTestDuration = 0, motorTestQueuedAt = 0;
int16_t lastCarLS = 0, lastCarRS = 0;
MotorSide mL = { PIN_AIN1_L, PIN_AIN2_L, PIN_PWM_L, 0, 0, 0, 0, false, 0, false, 0 };
MotorSide mR = { PIN_AIN1_R, PIN_AIN2_R, PIN_PWM_R, 0, 0, 0, 0, false, 0, false, 0 };

// ---- 新增：循迹状态（V19 原样）----
enum LineState { LS_FOLLOW, LS_PARKING, LS_CLEAR, LS_IDLE, LS_FAULT };
LineState lnState = LS_IDLE;
unsigned long lnStateEnterMs = 0, lnParkBlankUntilMs = 0;
PID lnPid = { KP_DEFAULT, KI_DEFAULT, KD_DEFAULT, 0, 0, 30.0f, false, 0.0f };
int lnBaseSpeed = SPEED_MED;
uint8_t lnSensorBits = 0, lnRawBits = 0;
bool lineTargetIsOne = false;   // 2026-10-18 实测: 本车传感器黑线=灭灯/bit0, 白地=亮灯/bit1
bool lineFlip = SENSOR_FLIP;
int8_t lineTurnSign = 1;
bool lineBusReady = false, lineWatch = false, lnResumePending = false;
bool lnParkingPending = false;
uint32_t lnParkingPendingAt = 0;
uint32_t lnReadErrors = 0;
uint8_t lnI2cStatus = 0;
const char *lnFaultReason = "NONE";
bool lnSensorOk = false;
int lnBlackCount = 0;
float lnPosition = 3.5f;
float lnError = 0;
int lnLSpeed = 0, lnRSpeed = 0;
bool lnLineLost = false, lnLostStopped = false;
unsigned long lnLostStartMs = 0;
unsigned long lnLastControlMs = 0, lnLastDebugMs = 0;

int clampInt(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

int clampPan(int us) { return clampInt(us, PAN_MIN_US, PAN_MAX_US); }
int clampTilt(int us) { return clampInt(us, TILT_MIN_US, TILT_MAX_US); }

// 直接写 LEDC 寄存器：core 3.x 的 ledcWrite() 实测每次阻塞约 40ms（等 PWM 周期），
// 直接写寄存器约 0.6us，控制环才能跑满 100Hz。
void setServoUs(uint8_t ch, int us) {
  us = clampInt(us, SERVO_MIN_US, SERVO_MAX_US);
  uint32_t duty = uint32_t(us) * 65535UL / 20000UL;
  LEDC.channel_group[0].channel[ch].duty.duty = duty << 4;  // 20bit 寄存器，低 4 位是小数
  LEDC.channel_group[0].channel[ch].conf1.duty_start = 1;   // 锁存下一周期生效
}

void applyServos() {
  panUs = clampPan(panUs);
  tiltUs = clampTilt(tiltUs);
  setServoUs(CH_PAN, panUs);
  setServoUs(CH_TILT, tiltUs);
}

void centerServos() {
  panUs = SERVO_MID_US;
  tiltUs = SERVO_MID_US;
  panVel = tiltVel = 0.0f;
  applyServos();
}

// ---------------- USB 命令 ----------------
void printStatus() {
  Serial.print("PAN="); Serial.print(panUs);
  Serial.print(" TILT="); Serial.print(tiltUs);
  Serial.print(" auto="); Serial.print(autoTrack ? 1 : 0);
  Serial.print(" state="); Serial.print(trackState);
  Serial.print(" frames="); Serial.print(frames);
  Serial.print(" bad="); Serial.print(badFrames);
  Serial.print(" last="); Serial.print(lastCx); Serial.print(","); Serial.print(lastCy);
  Serial.print(" bull="); Serial.print(lastBx); Serial.print(","); Serial.print(lastBy);
  Serial.print(" conf="); Serial.print(lastConf);
  Serial.print(" age="); Serial.print(millis() - lastFrameMs);
  Serial.print(" ctrlErr="); Serial.print(aimErrX, 1);
  Serial.print(","); Serial.print(aimErrY, 1);
  Serial.print(" bullLost="); Serial.println(aimBullLostFrames);
}

void help() {
  Serial.println("Commands: s1 <us> | s2 <us> | pan <d> | tilt <d> | sh | auto | manual | sweep | status");
  Serial.println("          uarttest  (short GPIO16<->17 to test UART2)");
  Serial.println("          pulse <pin> / ledcdump / s1api <us>  (PWM 自检)");
  Serial.println("          pid <kp> <ki> <kd> / pid2 ... / pidshow   (在线调 PID)");
  Serial.println("          trk <kp> <lead_s> <alpha> <beta>       (世界系跟踪器调参)");
  Serial.println("          trk <alpha> <lead_s> / smooth <tau_s>  (目标平滑/前视/速度低通)");
  Serial.println("          probe16   (sample GPIO16 for 10s; toggle K230 GPIO9)");
  Serial.println("          $KT,<seq>,<run>,<cx>,<cy>,<bx>,<by>,<conf>!  (模拟 K230 帧)");
  Serial.println("          kstop  (调试: TRACK 中只停 K230，用于验证靶心丢失自动复位)");
  helpExtra();
}

void sweepTest() {
  const bool old = autoTrack;
  autoTrack = false;
  Serial.println("SWEEP start");
  for (int u = 1200; u <= 1800; u += 100) { panUs = u; applyServos(); delay(250); }
  for (int u = 1800; u >= 1200; u -= 100) { panUs = u; applyServos(); delay(250); }
  for (int u = 1350; u <= 1650; u += 50)  { tiltUs = u; applyServos(); delay(250); }
  for (int u = 1650; u >= 1350; u -= 50)  { tiltUs = u; applyServos(); delay(250); }
  panUs = tiltUs = SERVO_MID_US;
  applyServos();
  autoTrack = old;
  Serial.println("SWEEP done, centered");
}

// ---------------- K230 协议 ----------------
void sendAck(unsigned long seq) {
  K230.print("$E,ACK,");
  K230.print(seq);
  K230.print(",");
  K230.print(trackState);
  K230.print(",");
  K230.print(panUs);
  K230.print(",");
  K230.print(tiltUs);
  K230.println("!");
}

void handleTargetLine(char *line, bool fromUsb) {
  strncpy(lastLine, line, sizeof(lastLine) - 1);
  lastLine[sizeof(lastLine) - 1] = '\0';

  unsigned long seq = 0;
  int run = 0, cx = -1, cy = -1, bx = -1, by = -1, conf = 0;
  int n = sscanf(line, "$KT,%lu,%d,%d,%d,%d,%d,%d",
                 &seq, &run, &cx, &cy, &bx, &by, &conf);
  if (n != 7) {
    ++badFrames;
    Serial.print("BAD KT (");
    Serial.print(n);
    Serial.print(" fields): ");
    Serial.println(line);
    return;
  }

  ++frames;
  lastCx = cx;
  lastCy = cy;
  lastBx = bx;
  lastBy = by;
  lastConf = clampInt(conf, 0, 1000);
  lastRun = (run != 0);
  lastFrameMs = millis();
  lastAckSeq = seq;
  // ---- 靶心丢失计数（updateTrack 据此判定打靶完成）----
  const bool bullOk = (run != 0) && bx >= 0 && by >= 0 && lastConf >= AIM_BULL_MIN_CONF;
  if (bullOk) {
    aimBullLostFrames = 0;
    aimLastBullMs = lastFrameMs;
    aimEverBull = true;
  } else if (gimbalMode == GIMBAL_TRACK) {
    ++aimBullLostFrames;
  }
  sendAck(seq);

  if (fromUsb) {
    // 逐帧回显太吵（20Hz），只在异常时提示
    if (n != 7) Serial.println("USB KT parse fail");
  }
}

void readLineFrom(Stream &s, char *buf, size_t cap, size_t &used,
                  void (*onLine)(char *, bool), bool fromUsb) {
  while (s.available()) {
    char c = static_cast<char>(s.read());
    if (c == '\r') continue;
    if (c == '!' || c == '\n') {
      if (used > 0) {
        buf[used] = '\0';
        onLine(buf, fromUsb);
        used = 0;
      }
    } else if (used < cap - 1) {
      buf[used++] = c;
    } else {
      used = 0;  // 超长丢弃
    }
  }
}

void onUsbLine(char *line, bool) {
  if (!strncmp(line, "$KT,", 4)) {
    handleTargetLine(line, true);
    return;
  }
  if (!strcmp(line, "kstop")) {
    // 调试：只给 K230 发 STOP，不改变 gimbalMode（用于验证靶心丢失自动复位）
    k230Cmd("STOP");
    return;
  }
  if (!strncmp(line, "pidshow", 7)) {
    Serial.printf("PAN kp=%.3f ki=%.4f kd=%.4f | TILT kp=%.3f ki=%.4f kd=%.4f\n",
                  panKp, panKi, panKd, tiltKp, tiltKi, tiltKd);
    return;
  }
  if (!strncmp(line, "pid2 ", 5)) {
    float kp, ki, kd;
    if (sscanf(line + 5, "%f %f %f", &kp, &ki, &kd) == 3) {
      tiltKp = kp; tiltKi = ki; tiltKd = kd; tiltInteg = 0;
      Serial.printf("TILT PID kp=%.3f ki=%.4f kd=%.4f\n", tiltKp, tiltKi, tiltKd);
    } else help();
    return;
  }
  if (!strncmp(line, "smooth ", 7)) {
    float t_;
    if (sscanf(line + 7, "%f", &t_) == 1) {
      velTau = t_;
      Serial.printf("SMOOTH tau=%.3fs\n", velTau);
    } else help();
    return;
  }
  if (!strncmp(line, "trk ", 4)) {
    float a, ld;
    int got = sscanf(line + 4, "%f %f", &a, &ld);
    if (got >= 1) {
      tgtAlpha = a;
      if (got >= 2) trkLead = ld;
      Serial.printf("TRK alpha=%.2f lead=%.3fs\n", tgtAlpha, trkLead);
    } else help();
    return;
  }
  if (!strncmp(line, "pid ", 4)) {
    float kp, ki, kd;
    if (sscanf(line + 4, "%f %f %f", &kp, &ki, &kd) == 3) {
      panKp = kp; panKi = ki; panKd = kd; panInteg = 0;
      Serial.printf("PAN PID kp=%.3f ki=%.4f kd=%.4f\n", panKp, panKi, panKd);
    } else help();
    return;
  }

  char cmd[16] = {0};
  int a = 0, b = 0;
  int n = sscanf(line, "%15s %d %d", cmd, &a, &b);
  if (n <= 0) return;

  if (!strcmp(cmd, "s1") && n >= 2) {
    panUs = clampPan(a); panVel = 0; applyServos();
    Serial.print("PAN="); Serial.println(panUs);
  } else if (!strcmp(cmd, "s2") && n >= 2) {
    tiltUs = clampTilt(a); tiltVel = 0; applyServos();
    Serial.print("TILT="); Serial.println(tiltUs);
  } else if (!strcmp(cmd, "pan") && n >= 2) {
    panUs = clampPan(panUs + a); panVel = 0; applyServos();
    Serial.print("PAN="); Serial.println(panUs);
  } else if (!strcmp(cmd, "tilt") && n >= 2) {
    tiltUs = clampTilt(tiltUs + a); tiltVel = 0; applyServos();
    Serial.print("TILT="); Serial.println(tiltUs);
  } else if (!strcmp(cmd, "sh")) {
    centerServos();
    Serial.println("CENTER 1500/1500");
  } else if (!strcmp(cmd, "auto")) {
    autoTrack = true;
    Serial.println("AUTO on");
  } else if (!strcmp(cmd, "manual")) {
    autoTrack = false;
    panVel = tiltVel = 0;
    Serial.println("AUTO off (manual)");
  } else if (!strcmp(cmd, "probe16")) {
    Serial.println("PROBE16 10s: GPIO16 as input; toggle K230 GPIO9 now");
    K230.end();
    pinMode(16, INPUT);
    unsigned long t0 = millis();
    while (millis() - t0 < 10000) {
      Serial.print(digitalRead(16));
      delay(50);
    }
    Serial.println();
    K230.begin(BAUD, SERIAL_8N1, BUS_RX, BUS_TX);
    Serial.println("PROBE16 done, UART2 reinit");
  } else if (!strcmp(cmd, "uarttest")) {
    Serial.println("UART2 TX test: $KT,777,... (short GPIO16<->17 to loop back)");
    K230.print("$KT,777,1,400,240,400,240,900!\n");
    K230.flush();
  } else if (!strcmp(cmd, "pulse") && n >= 2) {
    // 脉宽自检：把舵机信号跳线到 a 号脚，读回 PWM 高电平宽度
    int mn = 99999, mx = 0, avg = 0, cnt = 0;
    for (int i = 0; i < 10; i++) {
      unsigned long w = pulseIn(a, HIGH, 30000);
      if (w > 0) { mn = min(mn, (int)w); mx = max(mx, (int)w); avg += (int)w; cnt++; }
      delay(3);
    }
    Serial.print("PULSE pin="); Serial.print(a);
    Serial.print(" n="); Serial.print(cnt);
    Serial.print(" avg="); Serial.print(cnt ? avg / cnt : 0);
    Serial.print(" min="); Serial.print(mn == 99999 ? 0 : mn);
    Serial.print(" max="); Serial.println(mx);
  } else if (!strcmp(cmd, "ledcdump")) {
    Serial.print("LEDC.conf="); Serial.println((unsigned)LEDC.conf.val, HEX);
    for (int ch = 0; ch < 2; ch++) {
      Serial.print("CH"); Serial.print(ch);
      Serial.print(" conf0="); Serial.print((unsigned)LEDC.channel_group[0].channel[ch].conf0.val, HEX);
      Serial.print(" conf1="); Serial.print((unsigned)LEDC.channel_group[0].channel[ch].conf1.val, HEX);
      Serial.print(" duty="); Serial.println((unsigned)LEDC.channel_group[0].channel[ch].duty.val, HEX);
    }
    // 定时器是否在跑：隔 2ms 读两次计数
    for (int t = 0; t < 4; t++) {
      uint32_t v1 = LEDC.timer_group[0].timer[t].value.timer_cnt;
      delay(2);
      uint32_t v2 = LEDC.timer_group[0].timer[t].value.timer_cnt;
      Serial.print("TMR"); Serial.print(t);
      Serial.print(" conf="); Serial.print((unsigned)LEDC.timer_group[0].timer[t].conf.val, HEX);
      Serial.print(" cnt="); Serial.print(v1);
      Serial.print("->"); Serial.print(v2);
      Serial.print(" moving="); Serial.println((v2 != v1) ? 1 : 0);
    }
  } else if (!strcmp(cmd, "s1api") && n >= 2) {
    panUs = clampPan(a); panVel = 0;
    ledcWrite(PIN_PAN, uint32_t(panUs) * 65535UL / 20000UL);   // 老 API（会阻塞约 40ms）
    Serial.print("S1API PAN="); Serial.println(panUs);
  } else if (!strcmp(cmd, "sweep")) {
    sweepTest();
  } else if (!strcmp(cmd, "status")) {
    printStatus();
  } else if (handleCarLine(line)) {
    // 新增：V19 移植的小车 / 电机 / 循迹 / 模式命令
  } else {
    help();
  }
}

// ---------------- 云台跟踪（图像系 PID + 速度自适应前视）----------------
void updateTrack(uint32_t now) {
  static uint32_t lastCtrl = 0;
  uint32_t elapsed = now - lastCtrl;
  if (elapsed < AIM_CTRL_MS) return;
  if (elapsed > 100) elapsed = 100;   // 异常长的间隔不失真
  const float dt = elapsed / 1000.0f;
  lastCtrl = now;

  static bool tgtInit = false;
  static float tx = 0.0f, ty = 0.0f;
  static float vxF = 0.0f, vyF = 0.0f;
  static int hx[3] = {0, 0, 0}, hy[3] = {0, 0, 0};
  static uint8_t hn = 0, hi = 0;
  static uint8_t jumpN = 0;

  if (!autoTrack) {
    panVel = tiltVel = 0;
    tgtInit = false; pidInit = false; panInteg = tiltInteg = 0; hn = hi = 0; jumpN = 0;
    trackState = 0;
    return;
  }

  // ---- 靶心丢失 -> 判定打靶完成，复位云台（不动 carMode）----
  if (gimbalMode == GIMBAL_TRACK) {
    const bool lostFrames = aimEverBull && (aimBullLostFrames >= AIM_LOST_FRAMES);
    const bool lostMs = aimEverBull && aimLastBullMs && (now - aimLastBullMs > AIM_LOST_MS);
    const bool neverFound = !aimEverBull && (now - aimStartMs > AIM_START_GRACE_MS + AIM_LOST_MS);
    if (lostFrames || lostMs || neverFound) {
      Serial.print("AIM: bull lost -> DONE (frames="); Serial.print(aimBullLostFrames);
      Serial.print(" age="); Serial.print(aimLastBullMs ? (now - aimLastBullMs) : (now - aimStartMs));
      Serial.println("ms); gimbal NONE, K230 STOP");
      setGimbalMode(GIMBAL_NONE);
      return;
    }
  }

  const bool stale = (lastFrameMs == 0) || (now - lastFrameMs > TARGET_TIMEOUT_MS);
  if (stale) {
    panVel = tiltVel = 0;
    tgtInit = false; pidInit = false; panInteg = tiltInteg = 0; hn = hi = 0; jumpN = 0;
    trackState = 3;
    return;
  }
  if (!lastRun) {
    panVel = tiltVel = 0;
    tgtInit = false; pidInit = false; panInteg = tiltInteg = 0; hn = hi = 0; jumpN = 0;
    trackState = 0;
    return;
  }

  const int ax = (lastBx >= 0) ? lastBx : lastCx;
  const int ay = (lastBy >= 0) ? lastBy : lastCy;

  // ---- 3 点中值去尖峰 ----
  hx[hi] = ax; hy[hi] = ay; hi = (uint8_t)((hi + 1) % 3);
  if (hn < 3) hn++;
  int axm = ax, aym = ay;
  if (hn == 2) {
    axm = (hx[0] + hx[1]) / 2; aym = (hy[0] + hy[1]) / 2;
  } else if (hn >= 3) {
    const int a0 = hx[hi], a1 = hx[(hi + 1) % 3], a2 = hx[(hi + 2) % 3];
    const int b0 = hy[hi], b1 = hy[(hi + 1) % 3], b2 = hy[(hi + 2) % 3];
    axm = a0 + a1 + a2 - min(a0, min(a1, a2)) - max(a0, max(a1, a2));
    aym = b0 + b1 + b2 - min(b0, min(b1, b2)) - max(b0, max(b1, b2));
  }

  // ---- 低通 + 离群点抑制 ----
  const float prevTx = tx, prevTy = ty;
  if (!tgtInit) {
    tx = (float)axm; ty = (float)aym; tgtInit = true; jumpN = 0;
  } else if (abs((int)(axm - tx)) > TGT_JUMP_PX || abs((int)(aym - ty)) > TGT_JUMP_PX) {
    if (++jumpN >= TGT_JUMP_HOLD) { tx = (float)axm; ty = (float)aym; jumpN = 0; }
  } else {
    jumpN = 0;
    tx += ((float)axm - tx) * tgtAlpha;
    ty += ((float)aym - ty) * tgtAlpha;
  }

  // 滤波后的图像速度（前视用）
  const float vxNew = constrain((tx - prevTx) / dt, -3000.0f, 3000.0f);
  const float vyNew = constrain((ty - prevTy) / dt, -3000.0f, 3000.0f);
  vxF += (vxNew - vxF) * 0.35f;
  vyF += (vyNew - vyF) * 0.35f;

  // 速度自适应前视：慢速小前视（不抖），高速大前视（补延迟）
  const float leadF = constrain(fabsf(vxF) / 500.0f, 0.70f, 1.20f);
  const float leadT = constrain(fabsf(vyF) / 500.0f, 0.70f, 1.20f);
  // ---- 瞄准参考 = 画面中心：让靶心居中（PID/前视/低通结构不变）----
  const float errX = (tx + vxF * trkLead * leadF) - (float)CENTER_X;
  const float errY = (ty + vyF * trkLead * leadT) - (float)CENTER_Y;
  aimErrX = errX;
  aimErrY = errY;

  // ---- PID（输出角速度 us/s）----
  if (!pidInit) { panPrevE = errX; tiltPrevE = errY; pidInit = true; }
  panDFilt  += ((errX - panPrevE) / dt - panDFilt)  * PID_D_FILTER;
  tiltDFilt += ((errY - tiltPrevE) / dt - tiltDFilt) * PID_D_FILTER;
  panPrevE = errX; tiltPrevE = errY;

  if (fabsf(errX) > WBAND_PX) panInteg += errX * dt;
  if (fabsf(errY) > WBAND_PX) tiltInteg += errY * dt;
  if (panKi > 1e-5f) {
    const float maxI = PAN_MAX_RATE / panKi;
    panInteg = constrain(panInteg, -maxI, maxI);
  } else panInteg = 0.0f;
  if (tiltKi > 1e-5f) {
    const float maxI = TILT_MAX_RATE / tiltKi;
    tiltInteg = constrain(tiltInteg, -maxI, maxI);
  } else tiltInteg = 0.0f;

  float pidX = 0.0f, pidY = 0.0f;
  if (fabsf(errX) > WBAND_PX) pidX = panKp * errX + panKi * panInteg + panKd * panDFilt;
  if (fabsf(errY) > WBAND_PX) pidY = tiltKp * errY + tiltKi * tiltInteg + tiltKd * tiltDFilt;

  const float desiredPan  = constrain(-PAN_SIGN  * pidX, -PAN_MAX_RATE,  PAN_MAX_RATE);
  const float desiredTilt = constrain(-TILT_SIGN * pidY, -TILT_MAX_RATE, TILT_MAX_RATE);

  // 一阶低通：让速度指令平滑起停（丝滑感的主要来源）
  if (velTau > 0.001f) {
    const float a = 1.0f - expf(-dt / velTau);
    panVel  += (desiredPan  - panVel)  * a;
    tiltVel += (desiredTilt - tiltVel) * a;
  } else {
    panVel += constrain(desiredPan - panVel, -PAN_ACCEL * dt, PAN_ACCEL * dt);
    tiltVel += constrain(desiredTilt - tiltVel, -TILT_ACCEL * dt, TILT_ACCEL * dt);
  }
  panVel  = constrain(panVel,  -PAN_MAX_RATE,  PAN_MAX_RATE);
  tiltVel = constrain(tiltVel, -TILT_MAX_RATE, TILT_MAX_RATE);

  panUs = clampPan(panUs + int(panVel * dt));
  tiltUs = clampTilt(tiltUs + int(tiltVel * dt));
  applyServos();

  if (fabsf(errX) <= WBAND_PX && fabsf(errY) <= WBAND_PX) trackState = 2;
  else trackState = 1;
}


// ============================================================================
// 【新增】小车侧实现：模式 / 手柄 / 电机 / 循迹 / 命令层
// 全部移植自 ESP32_Finished_V19_LineMotor；云台仍由上面的 updateTrack() 负责，
// 手动微调只改 panUs/tiltUs 再调 applyServos()，不碰跟踪算法。
// ============================================================================

// ---------- 指挥 K230（R2 -> START, 离开跟踪 -> STOP） ----------
void k230Cmd(const char *verb) {
  K230.print("$E,"); K230.print(verb); K230.println("!");
  Serial.print("-> K230 $E,"); Serial.print(verb); Serial.println("!");
}

// ---------- 模式（两套并行状态） ----------
void setGimbalMode(uint8_t m) {
  if (m > GIMBAL_TRACK) return;
  gimbalMode = m;
  autoTrack = (m == GIMBAL_TRACK);        // 直接复用原有的跟踪开关
  panVel = tiltVel = 0.0f;
  if (m != GIMBAL_TRACK) trackState = 0;
  // 靶心丢失状态复位：R2 重新进入时从零开始
  aimBullLostFrames = 0;
  aimLastBullMs = 0;
  aimEverBull = false;
  aimStartMs = millis();
  Serial.print("GIMBAL: ");
  if (m == GIMBAL_TRACK)       Serial.println("TRACK (K230 $KT)");
  else if (m == GIMBAL_MANUAL) Serial.println("MANUAL (PS2 L-X / R-Y)");
  else                         Serial.println("NONE (保持当前位置)");
  // 让 K230 常驻代理开始/停止跑识别：R2 -> START，R1/无模式 -> STOP
  k230Cmd(m == GIMBAL_TRACK ? "START" : "STOP");
  if (m == GIMBAL_TRACK) {
    Serial.print("  $KT frames="); Serial.print(frames);
    Serial.print(" bad="); Serial.print(badFrames);
    if (lastFrameMs) {
      Serial.print(" age="); Serial.print(millis() - lastFrameMs);
      Serial.print("ms run="); Serial.println(lastRun ? 1 : 0);
    } else {
      Serial.println(" 还没收到过 $KT（查 K230 UART3 TX=32 -> ESP32 16 / RX=33 <- 17）");
    }
  }
}

void setCarMode(uint8_t m) {
  if (m > CAR_LINE) return;
  carMode = m;
  tbHardStop();                           // 切模式先断输出，必须重新 arm
  lastCarLS = lastCarRS = 0;
  if (m == CAR_LINE) {
    lnState = LS_FOLLOW; lnResetPID();
    Serial.println("CAR: LINE (auto follow)");
  } else if (m == CAR_MANUAL) {
    Serial.println("CAR: MANUAL (PS2 sticks)");
  } else {
    Serial.println("CAR: NONE");
  }
}

// ---------- 云台手动微调（复用原有 panUs/tiltUs + applyServos） ----------
void jogGimbal(uint8_t axis, int deltaUs) {
  if (axis == 0) panUs  = clampPan(panUs + deltaUs);
  else           tiltUs = clampTilt(tiltUs + deltaUs);
  panVel = tiltVel = 0.0f;                // 清掉跟踪残速，松手后不会继续滑
  applyServos();
}

void handleGimbal() {
  const uint32_t now = millis();
  if (now - lastGimbalSend < GIMBAL_POLL_MS) return;
  lastGimbalSend = now;
  const int16_t panD = ps2.lx() - 128;
  const int16_t tiltD = 128 - ps2.ry();
  if (abs(panD) > GIMBAL_DEADZONE)  jogGimbal(0, panD / GIMBAL_STEP_DIV);
  if (abs(tiltD) > GIMBAL_DEADZONE) jogGimbal(1, tiltD / GIMBAL_STEP_DIV);
}

// ---------- 手柄按键边沿 ----------
bool pressedEdge(BtnEdge &b) {
  const bool now = ps2.pressed(b.mask);
  const bool hit = now && !b.was;
  b.was = now;
  return hit;
}

void resetBtnEdges() {                    // 手柄掉帧: 要求全部松开后才重新认边沿
  BtnEdge *all[] = {&btnCar, &btnGimbal, &btnLine, &btnTrack, &btnSel, &btnStart,
                    &btnCross, &btnTri, &btnSq, &btnUp, &btnDown};
  for (BtnEdge *b : all) b->was = true;
}

void padService(uint32_t now) {           // 20ms 轮询一次
  static uint8_t ps2Streak = 0;
  if (now - lastPs2Poll < PS2_POLL_MS) return;
  lastPs2Poll = now;
  const bool receivedGoodPad = ps2.writeRead();
  if (receivedGoodPad) {
    if (ps2Streak < 250) ++ps2Streak;
    lastValidPs2 = now;
    const bool neutral = abs(int(ps2.ly()) - 128) < 15 &&
                         abs(int(ps2.ry()) - 128) < 15 &&
                         !ps2.pressed(PS2_L3) && !ps2.pressed(PS2_R3);
    if (neutral) { if (!padNeutralSince) padNeutralSince = now; }
    else padNeutralSince = 0;
  } else {
    ps2Streak = 0; padNeutralSince = 0; ++padBadFrames;
  }
  padIsValid = padKeepReady(padIsValid, ps2Streak, now - lastValidPs2);
  if (padIsValid) padEverSeen = true;
}

void padButtons(uint32_t now) {
  const bool handleOk = padIsValid && (now - lastValidPs2 <= PAD_FRESH_MS);
  if (!handleOk) {
    resetBtnEdges();
    if (motorArmed && motorArmFromPad && (carMode == CAR_MANUAL || carMode == CAR_LINE))
      tbHardStop();                        // 掉帧不自动恢复，必须重新按模式键
    if (carMode == CAR_MANUAL && (lastCarLS != 0 || lastCarRS != 0)) {
      tbHardStop(); lastCarLS = lastCarRS = 0;
    }
    // 失联兜底只在"确实接过手柄"之后生效：没接手柄时云台/电机由 USB 命令控制，
    // 不能被误判成掉线而关掉。
    if (padEverSeen && now - lastValidPs2 > 2000 && !padLostStopped) {
      padLostStopped = true;
      if (carMode == CAR_LINE) {
        lnState = LS_IDLE; tbHardStop();
        Serial.println("手柄失联 -> 巡线停车");
      } else {
        // 云台跟踪只依赖 K230 的 $KT, 不依赖手柄: 手柄掉线绝不能把跟踪关掉,
        // 否则表现就是"明明在 R2 模式, 云台却不动". 电机已经在上面断开了.
        Serial.println("手柄失联 -> 只断电机, 云台跟踪保持");
      }
    }
    return;
  }
  padLostStopped = false;

  // ---- 肩键直达模式（再按同一个键 = 重新 arm / 重置） ----
  if (pressedEdge(btnCar))   { setCarMode(CAR_MANUAL); motorArm(true); }
  if (pressedEdge(btnGimbal)) setGimbalMode(GIMBAL_MANUAL);
  if (pressedEdge(btnLine))  lineStart(true);
  if (pressedEdge(btnTrack)) setGimbalMode(GIMBAL_TRACK);
  if (pressedEdge(btnSel))   setCarMode(carMode == CAR_MANUAL ? CAR_LINE : CAR_MANUAL);

  // ---- ×: 任何模式都能急停 ----
  if (pressedEdge(btnCross)) {
    tbHardStop();
    lastCarLS = lastCarRS = 0;
    carEStop = true;
    panVel = tiltVel = 0.0f;
    if (carMode == CAR_LINE) lnState = LS_IDLE;
    Serial.println("急停 ×: 电机输出已切断");
  }

  // ---- △: 云台回中 ----
  if (pressedEdge(btnTri)) {
    centerServos();
    Serial.println("云台回中 1500/1500");
  }

  // ---- □: 跟踪下把跟踪滤波状态整块复位（等价于重新进入 TRACK） ----
  if (pressedEdge(btnSq) && gimbalMode == GIMBAL_TRACK) {
    autoTrack = false; delay(2); autoTrack = true;   // updateTrack 在关闭时清全部静态状态
    panVel = tiltVel = 0.0f;
    Serial.println("TRACK: 跟踪状态已复位");
  }

  // ---- 方向键: 只在巡线模式调速度档 ----
  if (carMode == CAR_LINE) {
    if (pressedEdge(btnUp)) {
      lnBaseSpeed = (lnBaseSpeed < SPEED_MED) ? SPEED_MED : SPEED_HIGH;
      Serial.print("巡线速度="); Serial.println(lnBaseSpeed);
    }
    if (pressedEdge(btnDown)) {
      lnBaseSpeed = (lnBaseSpeed > SPEED_MED) ? SPEED_MED : SPEED_LOW;
      Serial.print("巡线速度="); Serial.println(lnBaseSpeed);
    }
  }

  // ---- START: 云台回中 / 巡线停车 / 跟踪暂停恢复 ----
  if (pressedEdge(btnStart)) {
    if (gimbalMode == GIMBAL_MANUAL) {
      centerServos();
      Serial.println("云台回中 1500/1500");
    } else if (carMode == CAR_LINE) {
      lnState = LS_IDLE; tbDrive(0, 0);
      Serial.println("LINE: 手动停车 (go 或手柄 L2 恢复)");
    } else if (gimbalMode == GIMBAL_TRACK) {
      autoTrack = !autoTrack;
      panVel = tiltVel = 0.0f;
      Serial.println(autoTrack ? "TRACK: 恢复跟踪" : "TRACK: 暂停跟踪 (再按 START 恢复)");
    } else {
      carStop(); lastCarLS = lastCarRS = 0;
    }
  }
  // V19 语义: 按住 START / × 期间保持断开
  if (ps2.pressed(PS2_START) || ps2.pressed(PS2_CROSS)) {
    tbHardStop();
    if (carMode == CAR_LINE) lnState = LS_IDLE;
  }

  // ---- 并行: 小车和云台各跑各的 ----
  if (carMode == CAR_MANUAL) handleCar();
  if (gimbalMode == GIMBAL_MANUAL) handleGimbal();
}

void padPrint() {                          // 手柄诊断: 原始 9 字节 + 摇杆/按键
  char tmp[40];
  snprintf(tmp, sizeof(tmp), "%02X %02X %02X %02X %02X %02X %02X %02X %02X",
           ps2.byte(0), ps2.byte(1), ps2.byte(2), ps2.byte(3), ps2.byte(4),
           ps2.byte(5), ps2.byte(6), ps2.byte(7), ps2.byte(8));
  Serial.print("PS2["); Serial.print(tmp);
  Serial.print("] mode="); Serial.print(ps2.modeByte(), HEX);
  Serial.print(" valid="); Serial.print(padIsValid);
  Serial.print(" ever="); Serial.print(padEverSeen);
  Serial.print(" bad="); Serial.print(padBadFrames);
  Serial.print(" age="); Serial.print(millis() - lastValidPs2);
  Serial.print(" LX="); Serial.print(ps2.lx());
  Serial.print(" LY="); Serial.print(ps2.ly());
  Serial.print(" RX="); Serial.print(ps2.rx());
  Serial.print(" RY="); Serial.print(ps2.ry());
  Serial.print(" L3="); Serial.print(ps2.pressed(PS2_L3) ? 1 : 0);
  Serial.print(" R3="); Serial.print(ps2.pressed(PS2_R3) ? 1 : 0);
  Serial.print(" ST="); Serial.print(ps2.pressed(PS2_START) ? 1 : 0);
  Serial.print(" SEL="); Serial.println(ps2.pressed(PS2_SELECT) ? 1 : 0);
}

// ---------- 链路自诊断（1s 一次，只在出问题时提示） ----------
void linkDiag(uint32_t now) {
  static uint32_t last = 0;
  static bool noFrameWarned = false, notTrackWarned = false, noPadWarned = false;
  if (now - last < 1000) return;
  last = now;

  static uint32_t lastStartCmd = 0;
  const bool haveFrame = lastFrameMs && (now - lastFrameMs < 1000);
  if (gimbalMode == GIMBAL_TRACK) {
    notTrackWarned = false;
    if (!haveFrame) {
      // K230 可能晚启动 / 刚被复位 / 上次 STOP 了：每 3s 重喊一次 START
      if (now - lastStartCmd >= 3000) {
        lastStartCmd = now;
        k230Cmd("START");
      }
      if (!noFrameWarned) {
        noFrameWarned = true;
        Serial.print("TRACK 但收不到 $KT (age=");
        Serial.print(lastFrameMs ? now - lastFrameMs : 0);
        Serial.println("ms) -> 查 K230 UART3 TX=GPIO32 -> ESP32 GPIO16, RX=GPIO33 <- GPIO17, 共地, 115200");
      }
    } else {
      noFrameWarned = false;
    }
  } else if (haveFrame && lastRun && !notTrackWarned) {
    notTrackWarned = true;
    Serial.print("收到 $KT(run=1) 但云台不在 TRACK 模式 (gimbal=");
    Serial.print(gimbalMode);
    Serial.println("): 按手柄 R2, 或 USB 输入 gtrack");
  }

  if (!padEverSeen && now > 5000 && !noPadWarned) {
    noPadWarned = true;
    Serial.println("PS2 手柄一直没识别到 (pad=0): 查 DAT19/CMD18/ATT15/CLK23 与接收器供电；");
    Serial.println("  没有手柄不影响云台: 上电即进 TRACK, 也可用 USB gtrack / gmanual 切模式");
  }
  if (padEverSeen) noPadWarned = true;
}

// ---------- Motor control: bounded targets, reversal dwell, explicit arming ----------
static inline int16_t stepToward(int16_t value, int16_t goal, int16_t step) {
  if (value < goal) { int16_t n = value + step; return (n > goal || n < value) ? goal : n; }
  if (value > goal) { int16_t n = value - step; return (n < goal || n > value) ? goal : n; }
  return value;
}

void motorSetStandby(bool enabled) {
  motorStandbyCommand = enabled;
  digitalWrite(PIN_STBY, enabled ? HIGH : LOW);
}

void motorUpdateStandby() {
  motorSetStandby(motorEnableAllowed(motorPwmReady, motorArmed,
                 mL.target, mR.target, mL.applied, mR.applied));
}

void motorWrite(MotorSide &m, int16_t duty) {
  const int16_t d = constrain(duty, -MOTOR_MAX_DUTY, MOTOR_MAX_DUTY);
  const int8_t dir = (d > 0) ? 1 : ((d < 0) ? -1 : 0);
  if (dir != m.dir) {                 // 翻方向脚前 PWM 必须是 0
    ledcWrite(m.pwmPin, 0);
    if (dir > 0)      { digitalWrite(m.ain1, HIGH); digitalWrite(m.ain2, LOW);  }
    else if (dir < 0) { digitalWrite(m.ain1, LOW);  digitalWrite(m.ain2, HIGH); }
    else              { digitalWrite(m.ain1, LOW);  digitalWrite(m.ain2, LOW);  }
    m.dir = dir;
  }
  // IN1=IN2=LOW 且 PWM=HIGH 是 TB6612 真值表里的高阻停机
  ledcWrite(m.pwmPin, d == 0 ? 255 : uint8_t(d >= 0 ? d : -d));
  m.applied = d;
}

bool motorArm(bool fromPad) {
  tbHardStop();                       // 每次 arm 先清掉旧输出
  if (!motorPwmReady) { Serial.println("ARM rejected: 电机 PWM 初始化失败"); return false; }
  if (carMode != CAR_MANUAL && carMode != CAR_LINE) {
    Serial.println("ARM rejected: 先按 L1(小车) 或 L2(循迹) 选小车模式");
    return false;
  }
  const uint32_t now = millis();
  if (fromPad && (!padIsValid || now - lastValidPs2 > PAD_FRESH_MS ||
      padNeutralSince == 0 || now - padNeutralSince < PAD_NEUTRAL_HOLD_MS ||
      ps2.pressed(PS2_START) || ps2.pressed(PS2_CROSS))) {
    Serial.println("ARM rejected: 松开 STOP/×, 摇杆回中 500ms 后再按模式键");
    return false;
  }
  if (carMode == CAR_LINE) {
    lnReadLineSensors();
    if (!lnSensorOk || lnBlackCount == 0) {
      Serial.println("ARM rejected: 循迹传感器必须读到 OK 且能看到线");
      return false;
    }
  }
  motorArmed = true; carEStop = false; motorArmFromPad = fromPad;
  lineWatch = false;
  Serial.println("MOTOR ARMED: STBY 只在有输出请求时使能");
  return true;
}

void tbDrive(int16_t lSpeed, int16_t rSpeed) {
  if (!motorArmed || !motorPwmReady || carEStop) { mL.target = mR.target = 0; return; }
  if (TB_INV_L) lSpeed = -lSpeed;
  if (TB_INV_R) rSpeed = -rSpeed;
  for (uint8_t i = 0; i < 2; ++i) {
    MotorSide &m = (i == 0) ? mL : mR;
    if (m.brake) {                    // 刹车途中来了新指令: 先把桥臂放掉
      m.brake = false;
      m.dir = -1;
      motorWrite(m, 0);
      m.zeroAt = millis();
    }
  }
  mL.target = constrain(lSpeed, -MOTOR_MAX_DUTY, MOTOR_MAX_DUTY);
  mR.target = constrain(rSpeed, -MOTOR_MAX_DUTY, MOTOR_MAX_DUTY);
}

void tbHardStop() {                              // 立刻断输出: 急停 / 失联
  motorSetStandby(false);                        // 先关 STBY 再动 PWM/方向
  motorArmed = false; carEStop = true; motorTestActive = false; motorArmFromPad = false;
  motorTestOutputStarted = false;
  lastCarLS = lastCarRS = 0;
  const uint32_t now = millis();
  for (uint8_t i = 0; i < 2; ++i) {
    MotorSide &m = (i == 0) ? mL : mR;
    const bool wasDriven = m.applied != 0 || m.brake;
    m.target = 0; m.applied = 0; m.dir = 0; m.brake = false; m.reversing = false;
    if (wasDriven) m.zeroAt = now;               // 换向静置计时保留
    ledcWrite(m.pwmPin, 0);
    digitalWrite(m.ain1, LOW); digitalWrite(m.ain2, LOW);
  }
}

void tbBrake() {
  // 不做短路刹车: 电驱未标定, 直接高阻停机
  tbHardStop();
}

void motorServiceSide(MotorSide &m, uint32_t now, float dt) {
  m.brake = false;
  int16_t tgt = m.target;
  int16_t app = m.applied;
  const int16_t step = max(int16_t(1), int16_t(MOTOR_SLEW_PER_S * dt));

  const int8_t wantDir = (tgt > 0) ? 1 : ((tgt < 0) ? -1 : 0);
  m.reversing = motorReverseWait(wantDir, m.lastDir, app, now, m.zeroAt,
                                 MOTOR_REVERSE_DEAD_MS);
  if (m.reversing) tgt = 0;

  if (tgt == 0)                    app = stepToward(app, 0, step);
  else if (app == 0)               { if (!m.reversing) app = stepToward(app, tgt, step); }
  else if ((app > 0) != (tgt > 0)) app = stepToward(app, 0, step);
  else                             app = stepToward(app, tgt, step);

  const int16_t prev = m.applied;
  motorWrite(m, app);
  if (app == 0 && prev != 0) m.zeroAt = now;     // 刚归零, 开始计静置时间
  if (app != 0) m.lastDir = (app > 0) ? 1 : -1;
}

void motorService(uint32_t now) {
  if (!motorArmed || !motorPwmReady) { motorSetStandby(false); return; }
  static uint32_t lastMs = 0;
  if (lastMs && now - lastMs > 100 && (mL.applied != 0 || mR.applied != 0)) {
    tbHardStop(); lastMs = now;
    Serial.println("MOTOR DISARMED: service gap >100ms"); return;
  }
  if (now - lastMs < MOTOR_CTRL_MS) return;
  float dt = (now - lastMs) / 1000.0f;
  if (dt > 0.05f) dt = 0.05f;                    // 长时间阻塞后不许一步跳满
  lastMs = now;
  motorUpdateStandby();                          // 先使能再给斜坡
  motorServiceSide(mL, now, dt);
  motorServiceSide(mR, now, dt);
  motorUpdateStandby();                          // 都停了再回待机
}

void scheduleDriveTest(int16_t l, int16_t r, uint32_t ms) {
  if (!motorArmed || carEStop || !motorPwmReady || carMode != CAR_MANUAL) {
    Serial.println("TEST rejected: 先 manual, 再 arm"); return;
  }
  if (ms < 20 || ms > 1500) { Serial.println("TEST duration 20..1500ms"); return; }
  if (l == 0 && r == 0) { tbHardStop(); return; }
  motorTestQueuedAt = millis(); motorTestStart = 0;
  motorTestDuration = ms; motorTestActive = true; motorTestOutputStarted = false;
  motorArmFromPad = false;
  tbDrive(l, r);
  Serial.println("MOTOR TEST scheduled; cs/START/× 取消");
}

void motorTestService(uint32_t now) {
  if (!motorTestActive) return;
  if (!motorTestOutputStarted) {
    if (mL.applied != 0 || mR.applied != 0) {
      motorTestOutputStarted = true; motorTestStart = now;
      Serial.println("MOTOR TEST: 输出已建立, 开始计时");
    } else if (now - motorTestQueuedAt >= MOTOR_TEST_START_TIMEOUT_MS) {
      tbHardStop(); Serial.println("MOTOR TEST aborted: 超时没有输出");
    }
    return;
  }
  if (now - motorTestStart >= motorTestDuration) {
    tbHardStop(); Serial.println("MOTOR TEST complete; DISARMED");
  }
}

void motorPinsDiagnostic(int side, int direction) {
  tbHardStop();
  const uint8_t in1 = side == 0 ? PIN_AIN1_L : PIN_AIN1_R;
  const uint8_t in2 = side == 0 ? PIN_AIN2_L : PIN_AIN2_R;
  digitalWrite(in1, direction > 0 ? HIGH : LOW);
  digitalWrite(in2, direction < 0 ? HIGH : LOW);
  Serial.printf("PINS %s: GPIO%u=%d GPIO%u=%d STBY25=LOW; 电机已断开\n",
                side == 0 ? "LEFT" : "RIGHT", in1, direction > 0, in2, direction < 0);
}

void motorPrint() {
  Serial.print("MOTOR armed="); Serial.print(motorArmed);
  Serial.print(" pwm_ready="); Serial.print(motorPwmReady);
  Serial.print(" STBY25_cmd="); Serial.print(motorStandbyCommand);
  Serial.print(" max="); Serial.print(MOTOR_MAX_DUTY);
  Serial.print(" slew="); Serial.print(MOTOR_SLEW_PER_S);
  Serial.print("/s dead="); Serial.print(MOTOR_REVERSE_DEAD_MS); Serial.println("ms");
  Serial.print("  L tgt="); Serial.print(mL.target);
  Serial.print(" out="); Serial.print(mL.applied);
  Serial.print(" dir="); Serial.print(mL.dir);
  Serial.print(" rev="); Serial.print(mL.reversing ? 1 : 0);
  Serial.print(" last="); Serial.println(mL.lastDir);
  Serial.print("  R tgt="); Serial.print(mR.target);
  Serial.print(" out="); Serial.print(mR.applied);
  Serial.print(" dir="); Serial.print(mR.dir);
  Serial.print(" rev="); Serial.print(mR.reversing ? 1 : 0);
  Serial.print(" last="); Serial.println(mR.lastDir);
  Serial.printf("TEST active=%d output_started=%d duration=%lu ms\n",
                motorTestActive, motorTestOutputStarted, (unsigned long)motorTestDuration);
}

void carStop() { tbHardStop(); }

int16_t stickToSpeed(int16_t stickY) { return int16_t((128 - stickY) * 2.0f); }

void handleCar() {
  if (!motorArmed || !motorArmFromPad || carEStop || motorTestActive) return;
  int16_t lS = stickToSpeed(ps2.ly()), rS = stickToSpeed(ps2.ry());
  if (abs(lS) < CAR_SPEED_DEADZONE) lS = 0;
  if (abs(rS) < CAR_SPEED_DEADZONE) rS = 0;
  lS = constrain(lS, -CAR_MAX_SPEED, CAR_MAX_SPEED);
  rS = constrain(rS, -CAR_MAX_SPEED, CAR_MAX_SPEED);
  if (ps2.pressed(PS2_L3)) { lS = -CAR_SPIN_SPEED; rS = CAR_SPIN_SPEED; }
  else if (ps2.pressed(PS2_R3)) { lS = CAR_SPIN_SPEED; rS = -CAR_SPIN_SPEED; }
  if (lS != lastCarLS || rS != lastCarRS) {
    lastCarLS = lS; lastCarRS = rS; tbDrive(lS, rS);
  }
}

// ---------- 循迹（V19 原样移植；上报/继续改走 K230 串口） ----------
void lnResetPID() {
  lnPid.integral = 0; lnPid.prevError = 0; lnPid.hasPrev = false; lnPid.derivative = 0;
}

void lnReadLineSensors() {
  lnSensorOk = false; lnSensorBits = 0; lnBlackCount = 0; lnPosition = 3.5f;
  if (!lineBusReady) { lnI2cStatus = 254; ++lnReadErrors; return; }
  Wire.beginTransmission(I2C_ADDR); Wire.write(I2C_REG_STATE);
  lnI2cStatus = Wire.endTransmission(true);      // 先选寄存器再读（带 STOP）
  if (lnI2cStatus != 0) { ++lnReadErrors; return; }
  if (Wire.requestFrom(I2C_ADDR, uint8_t(1), true) != 1) {
    lnI2cStatus = 253; ++lnReadErrors;
    while (Wire.available()) Wire.read();
    return;
  }
  const int value = Wire.read();
  if (value < 0 || value > 255) { lnI2cStatus = 252; ++lnReadErrors; return; }
  lnRawBits = uint8_t(value);
  lnSensorBits = lineTargetIsOne ? lnRawBits : uint8_t(~lnRawBits);
  int sum = 0;
  for (int i = 0; i < 8; ++i) if (lnSensorBits & (1u << i)) {
    sum += lineFlip ? (7 - i) : i; ++lnBlackCount;
  }
  if (lnBlackCount) lnPosition = float(sum) / lnBlackCount;
  lnSensorOk = true; lnI2cStatus = 0;
}

float lnPidCompute(PID &p, float error, float dt) {
  dt = constrain(dt, 0.005f, 0.1f);
  if (!p.hasPrev) {
    p.hasPrev = true; p.prevError = error; p.integral = 0; p.derivative = 0;
    return p.kp * error;
  }
  p.integral = constrain(p.integral + error * dt, -p.integralMax, p.integralMax);
  const float rawDerivative = (error - p.prevError) / dt;
  p.derivative += 0.25f * (rawDerivative - p.derivative);
  p.prevError = error;
  return p.kp * error + p.ki * p.integral + p.kd * p.derivative;
}

void lnEnterParking() {
  lnState = LS_PARKING;
  lnStateEnterMs = millis();
  tbBrake();
  lnResetPID();
  Serial.println(RSP_PARKED);
  K230.println(RSP_PARKED);                 // 上报"已停稳", 等 K230 的 DONE
  Serial.println("检测到停车标记 -> 停车, 等待 DONE");
}

void lnResumeFromParking() {
  // 外部消息不能绕过电机联锁: 只置 pending, 仍要显式 arm
  lnResumePending = true;
  Serial.println("LINE: 收到继续请求; 仍需 L2/go 显式使能");
}

void lnHandleParkingDetection() {
  const uint32_t now = millis();
  if (lnParkBlankUntilMs && int32_t(now - uint32_t(lnParkBlankUntilMs)) < 0) return;
  lnParkBlankUntilMs = 0;
  if (lnBlackCount >= PARKING_MIN_BLACK) {
    if (!lnParkingPending) { lnParkingPending = true; lnParkingPendingAt = now; }
    else if (now - lnParkingPendingAt >= PARKING_DEBOUNCE_MS) {
      lnParkingPending = false; lnEnterParking();
    }
  } else lnParkingPending = false;
}

void lnFault(const char *reason) {
  const bool report = lnState != LS_FAULT || lnFaultReason != reason;
  tbHardStop(); lnState = LS_FAULT; lnFaultReason = reason;
  lnLSpeed = lnRSpeed = 0; lnResetPID();
  if (report) { Serial.print("LINE FAULT: "); Serial.println(reason); }
}

void lnFollowLine(float dt) {
  if (!lnSensorOk) { lnFault("I2C_READ_FAILED"); return; }
  // 误差按传感器间距(-3.5..+3.5), 不归一化
  float error = (lnPosition - 3.5f) * lineTurnSign;
  int speed = min(lnBaseSpeed, int(MOTOR_MAX_DUTY));
  float correction;
  if (!lnBlackCount) {
    if (!lnLineLost) {
      lnLineLost = true; lnLostStartMs = millis(); lnResetPID(); lnLostStopped = false;
    }
    if (uint32_t(millis() - lnLostStartMs) >= LOST_TIMEOUT_MS) {
      lnLostStopped = true; lnFault("LINE_LOST_TIMEOUT"); return;
    }
    speed = int(speed * LOST_SPEED_FACTOR);
    error = lnError; correction = lnPid.kp * error;
  } else {
    lnLineLost = false; lnLostStopped = false;
    correction = lnPidCompute(lnPid, error, dt);
  }
  correction = constrain(correction, -float(speed), float(speed));
  lnError = error;
  lnLSpeed = constrain(speed + int(correction), 0, int(MOTOR_MAX_DUTY));
  lnRSpeed = constrain(speed - int(correction), 0, int(MOTOR_MAX_DUTY));
  tbDrive(lnLSpeed, lnRSpeed);
}

void lnRunControl(float dt) {
  lnReadLineSensors();
  if (!lnSensorOk) { lnFault("I2C_READ_FAILED"); return; }
  if (!motorArmed || carEStop) { lnLSpeed = lnRSpeed = 0; return; }
  switch (lnState) {
    case LS_FOLLOW:
      lnHandleParkingDetection();
      if (lnState == LS_FOLLOW) lnFollowLine(dt);
      break;
    case LS_PARKING: break;
    case LS_CLEAR: {
      const uint32_t elapsed = millis() - lnStateEnterMs;
      if (lnBlackCount < PARKING_MIN_BLACK && elapsed >= CLEAR_MARKER_MIN_MS) {
        lnState = LS_FOLLOW; lnResetPID(); lnParkBlankUntilMs = millis() + PARK_BLANK_MS;
        Serial.println("LINE: marker cleared; FOLLOW");
      } else if (elapsed >= CLEAR_MARKER_MAX_MS) lnFault("MARKER_CLEAR_TIMEOUT");
      else tbDrive(lnBaseSpeed, lnBaseSpeed);
      break;
    }
    case LS_IDLE: case LS_FAULT: tbDrive(0, 0); break;
  }
}

bool lineStart(bool fromPad) {
  const bool leaveMarker = carMode == CAR_LINE && (lnState == LS_PARKING || lnResumePending);
  setCarMode(CAR_LINE);
  if (!motorArm(fromPad)) { lnPrintStatus(); return false; }
  lineWatch = false; lnFaultReason = "NONE";
  lnLineLost = false; lnLostStopped = false; lnParkingPending = false;
  lnResumePending = false; lnParkBlankUntilMs = 0;
  lnResetPID(); lnLastControlMs = millis();
  lnState = leaveMarker ? LS_CLEAR : LS_FOLLOW; lnStateEnterMs = millis();
  Serial.println(leaveMarker ? "LINE ARMED: 先越线" : "LINE ARMED: FOLLOW");
  return true;
}

void lineWatchService(uint32_t now) {
  static uint32_t last = 0;
  if (!lineWatch || now - last < 200) return;
  last = now; lnReadLineSensors(); lnPrintStatus();
}

void lineService(uint32_t now) {
  if (carMode == CAR_LINE && now - lnLastControlMs >= LINE_CTRL_MS) {
    const float lineDt = constrain(float(uint32_t(now - lnLastControlMs)) / 1000.0f, 0.005f, 0.1f);
    lnLastControlMs = now;
    lnRunControl(lineDt);
  }
  lineWatchService(now);
}

void lnPrintStatus() {
  Serial.print("LINE 状态=");
  switch (lnState) {
    case LS_FOLLOW:  Serial.print("FOLLOW"); break;
    case LS_PARKING: Serial.print("PARKING"); break;
    case LS_CLEAR:   Serial.print("CLEAR"); break;
    case LS_IDLE:    Serial.print("IDLE"); break;
    case LS_FAULT:   Serial.print("FAULT"); break;
  }
  Serial.printf(" armed=%d i2c=%s status=%u raw=0x%02X mask=0x%02X count=%d pos=%.2f fault=%s target1=%d flip=%d turn=%d errors=%lu",
                motorArmed, lnSensorOk ? "OK" : "FAIL", lnI2cStatus,
                lnRawBits, lnSensorBits, lnBlackCount, lnPosition, lnFaultReason,
                lineTargetIsOne, lineFlip, lineTurnSign, (unsigned long)lnReadErrors);
  Serial.print(" 速度="); Serial.print(lnBaseSpeed);
  Serial.print(" Kp="); Serial.print(lnPid.kp);
  Serial.print(" Ki="); Serial.print(lnPid.ki);
  Serial.print(" Kd="); Serial.println(lnPid.kd);
}

// ---------- 新增命令层（原有云台命令先匹配, 这里只接新的） ----------
void modeStatus() {
  Serial.print("MODE car=");
  Serial.print(carMode == CAR_MANUAL ? "MANUAL" : (carMode == CAR_LINE ? "LINE" : "NONE"));
  Serial.print(" gimbal=");
  Serial.print(gimbalMode == GIMBAL_TRACK ? "TRACK" : (gimbalMode == GIMBAL_MANUAL ? "MANUAL" : "NONE"));
  Serial.print(" armed="); Serial.print(motorArmed);
  Serial.print(" estop="); Serial.print(carEStop);
  Serial.print(" pad="); Serial.print(padIsValid);
  Serial.print(" pad_age="); Serial.print(millis() - lastValidPs2);
  Serial.print(" pan="); Serial.print(panUs);
  Serial.print(" tilt="); Serial.print(tiltUs);
  Serial.print(" auto="); Serial.println(autoTrack);
}

void carStatusLine(uint32_t now) {
  static uint32_t last = 0;
  if (now - last < 1000) return;
  last = now;
  Serial.print("[CAR] car=");
  Serial.print(carMode == CAR_MANUAL ? "MANUAL" : (carMode == CAR_LINE ? "LINE" : "NONE"));
  Serial.print(" gimbal=");
  Serial.print(gimbalMode == GIMBAL_TRACK ? "TRACK" : (gimbalMode == GIMBAL_MANUAL ? "MANUAL" : "NONE"));
  Serial.print(" armed="); Serial.print(motorArmed);
  Serial.print(" pad="); Serial.print(padIsValid);
  Serial.print(" L="); Serial.print(mL.applied);
  Serial.print(" R="); Serial.print(mR.applied);
  Serial.print(" auto="); Serial.print(autoTrack);
  Serial.print(" ln="); Serial.print(lnSensorOk ? lnBlackCount : -1);
  Serial.print(" pad_bad="); Serial.print(padBadFrames);
  Serial.println();
}

void helpExtra() {
  Serial.println("-- 新增(V19 补齐): 手柄 --");
  Serial.println("PS2: L1=小车手动+arm  L2=循迹+arm  R1=云台手动  R2=云台跟踪(K230)");
  Serial.println("     SELECT=小车手动/循迹互切  X=急停  △=云台回中  START=停车/云台中位/跟踪暂停");
  Serial.println("-- 新增(V19 补齐): USB 命令 --");
  Serial.println("模式: cmanual | cline | cnone | gmanual(gm) | gtrack | gnone | mode");
  Serial.println("电机: arm | disarm/cs | stop | motor | ch <0左|1右> <-120..120> <20..1500ms> | cf | cb");
  Serial.println("      pwm <40..120>(先 disarm) | pins <0|1> <-1|0|1>(诊断, STBY 保持 LOW)");
  Serial.println("循迹: line | go | park | s | 1/2/3(速度) | kp/ki/kd <值> | linewatch on|off");
  Serial.println("      linepol 0|1 | lineflip 0|1 | lineturn -1|1");
  Serial.println("其他: home(舵机回中) | kt(=status) | sniff [ms](偷看 K230 串口原始字节, 先断电机)");
  Serial.println("       pad/ps2(手柄原始 9 字节与摇杆/按键, 查接线用)");
}

bool handleCarLine(const char *line) {
  if (!strcmp(line, "arm")) { motorArm(false); return true; }
  if (!strcmp(line, "disarm") || !strcmp(line, "cs")) {
    tbHardStop(); if (carMode == CAR_LINE) lnState = LS_IDLE;
    Serial.println("MOTOR DISARMED"); return true;
  }
  if (!strcmp(line, "stop")) {
    tbHardStop(); if (carMode == CAR_LINE) { lnState = LS_IDLE; tbDrive(0, 0); }
    panVel = tiltVel = 0.0f;
    Serial.println("STOP: 电机断开, 云台保持当前角度"); return true;
  }
  if (!strcmp(line, "mode")) { modeStatus(); return true; }
  if (!strcmp(line, "motor")) { motorPrint(); return true; }
  if (!strcmp(line, "pad") || !strcmp(line, "ps2")) { padPrint(); return true; }
  if (!strcmp(line, "kt")) { printStatus(); return true; }
  if (!strncmp(line, "pins ", 5)) {
    int side = -1, direction = 9; char extra;
    if (sscanf(line + 5, "%d %d %c", &side, &direction, &extra) == 2 &&
        (side == 0 || side == 1) && direction >= -1 && direction <= 1)
      motorPinsDiagnostic(side, direction);
    else Serial.println("pins <0=left|1=right> <-1|0|1>; STBY 保持 LOW");
    return true;
  }
  if (!strncmp(line, "ch ", 3)) {
    int side = 0, spd = 0, t = 0;
    if (sscanf(line + 3, "%d %d %d", &side, &spd, &t) == 3 &&
        (side == 0 || side == 1) && spd >= -MOTOR_MAX_DUTY && spd <= MOTOR_MAX_DUTY &&
        t >= 20 && t <= 1500) {
      scheduleDriveTest(side == 0 ? spd : 0, side == 1 ? spd : 0, uint32_t(t));
    } else Serial.println("usage: ch <0=left|1=right> <-120..120> <20..1500ms>; 先 arm");
    return true;
  }
  if (!strncmp(line, "pwm ", 4)) {
    const int v = atoi(line + 4);
    if (v >= 40 && v <= MOTOR_HARD_CAP && !motorArmed) {
      MOTOR_MAX_DUTY = int16_t(v);
      Serial.print("电机最大PWM="); Serial.print(MOTOR_MAX_DUTY);
      Serial.println(" (只是软件占空比上限, 不是电流限制)");
    } else Serial.println("usage: 先 disarm, 再 pwm 40..120");
    return true;
  }
  if (!strcmp(line, "cf")) { scheduleDriveTest(80, 80, 500); return true; }
  if (!strcmp(line, "cb")) { scheduleDriveTest(-80, -80, 500); return true; }

  // ---- 模式 ----
  if (!strcmp(line, "cmanual")) { setCarMode(CAR_MANUAL); return true; }
  if (!strcmp(line, "cline")) {
    setCarMode(CAR_LINE); lnReadLineSensors(); lnPrintStatus();
    Serial.println("LINE selected, DISARMED; 用 go 或手柄 L2 使能"); return true;
  }
  if (!strcmp(line, "cnone")) { setCarMode(CAR_NONE); return true; }
  if (!strcmp(line, "gmanual") || !strcmp(line, "gm")) { setGimbalMode(GIMBAL_MANUAL); return true; }
  if (!strcmp(line, "gtrack") || !strcmp(line, "vision") || !strcmp(line, "vt")) {
    setGimbalMode(GIMBAL_TRACK); return true;
  }
  if (!strcmp(line, "gnone")) { setGimbalMode(GIMBAL_NONE); return true; }
  if (!strcmp(line, "home")) {
    centerServos();
    Serial.println("云台回中 1500/1500"); return true;
  }

  // ---- 巡线 ----
  if (!strcmp(line, "line")) {
    setCarMode(CAR_LINE); lnReadLineSensors(); lnPrintStatus();
    Serial.println("LINE selected, DISARMED; 用 go 或手柄 L2 使能"); return true;
  }
  if (!strcmp(line, "go")) {
    if (carMode == CAR_LINE) lineStart(false);
    else Serial.println("当前不是巡线模式 (先输入 line)");
    return true;
  }
  if (!strcmp(line, "park")) {
    if (carMode == CAR_LINE && lnState == LS_FOLLOW) lnEnterParking();
    else Serial.println("仅巡线模式 FOLLOW 时可用");
    return true;
  }
  if (!strcmp(line, "s")) { lnReadLineSensors(); lnPrintStatus(); return true; }
  if (!strcmp(line, "1")) { lnBaseSpeed = SPEED_LOW;  Serial.print("速度=低速 ");  Serial.println(lnBaseSpeed); return true; }
  if (!strcmp(line, "2")) { lnBaseSpeed = SPEED_MED;  Serial.print("速度=中速 ");  Serial.println(lnBaseSpeed); return true; }
  if (!strcmp(line, "3")) { lnBaseSpeed = SPEED_HIGH; Serial.print("速度=高速 ");  Serial.println(lnBaseSpeed); return true; }
  if (!strncmp(line, "kp ", 3)) { lnPid.kp = atof(line + 3); Serial.print("Kp="); Serial.println(lnPid.kp); return true; }
  if (!strncmp(line, "ki ", 3)) { lnPid.ki = atof(line + 3); Serial.print("Ki="); Serial.println(lnPid.ki); return true; }
  if (!strncmp(line, "kd ", 3)) { lnPid.kd = atof(line + 3); Serial.print("Kd="); Serial.println(lnPid.kd); return true; }
  if (!strcmp(line, "linewatch on"))  { tbHardStop(); lineWatch = true;  Serial.println("LINEWATCH: 电机已断开"); return true; }
  if (!strcmp(line, "linewatch off")) { lineWatch = false; return true; }
  if (!strncmp(line, "linepol ", 8) || !strncmp(line, "lineflip ", 9) || !strncmp(line, "lineturn ", 9)) {
    int value = 99; char extra;
    const bool pol = !strncmp(line, "linepol ", 8);
    const bool turn = !strncmp(line, "lineturn ", 9);
    const char *arg = line + (pol ? 8 : 9);
    if (sscanf(arg, "%d %c", &value, &extra) == 1 &&
        (turn ? (value == -1 || value == 1) : (value == 0 || value == 1))) {
      tbHardStop(); lnState = LS_IDLE;
      if (pol) lineTargetIsOne = value; else if (turn) lineTurnSign = value; else lineFlip = value;
      lnResetPID(); lnReadLineSensors(); lnPrintStatus();
    } else Serial.println("linepol 0|1; lineflip 0|1; lineturn -1|1 (都会断电机)");
    return true;
  }

  // ---- 诊断: 偷看 K230 串口原始字节（阻塞, 期间不解析 $KT） ----
  if (!strncmp(line, "sniff", 5)) {
    tbHardStop();                                   // 诊断期间不允许电机带电
    unsigned long ms = 2000;
    if (sscanf(line + 5, "%lu", &ms) != 1 || ms == 0) ms = 2000;
    if (ms > 10000) ms = 10000;
    Serial.print("SNIFF "); Serial.print(ms); Serial.println("ms on K230 UART2 (不可见字符用 . 代替)...");
    const uint32_t t0 = millis(); uint32_t total = 0; char buf[65]; uint8_t n = 0;
    while (millis() - t0 < ms) {
      while (K230.available()) {
        const char c = static_cast<char>(K230.read()); ++total;
        buf[n++] = (c >= 32 && c < 127) ? c : '.';
        if (n == 64) { buf[64] = '\0'; Serial.println(buf); n = 0; }
      }
      delay(1);
    }
    if (n) { buf[n] = '\0'; Serial.println(buf); }
    Serial.print("SNIFF done, bytes="); Serial.println(total);
    return true;
  }
  return false;
}

// ---------- K230 串口行分发（原有 handleTargetLine 仍负责 $KT） ----------
void onK230Line(char *line, bool fromUsb) {
  if (!strncmp(line, "$KT,", 4)) { handleTargetLine(line, fromUsb); return; }
  if (!strcmp(line, CMD_DONE)) { lnResumeFromParking(); return; }
  if (!strncmp(line, "$E,K230,", 8)) {              // K230 常驻代理的回执
    Serial.print("K230: "); Serial.println(line + 8);
    // K230 刚上电/刚复位: 它报 READY 时如果我们在跟踪, 立刻补一条 START
    if (!strcmp(line + 8, "READY") && gimbalMode == GIMBAL_TRACK) k230Cmd("START");
    return;
  }
  if (!strncmp(line, "$E,", 3)) return;             // 其它 $E 帧(ACK/READY)忽略
  static uint32_t lastNote = 0;
  if (millis() - lastNote >= 2000) {
    lastNote = millis();
    Serial.print("K230 收到未知帧: "); Serial.println(line);
  }
}

// ---------------- setup / loop ----------------
void setup() {
  Serial.begin(BAUD);
  K230.begin(BAUD, SERIAL_8N1, BUS_RX, BUS_TX);

  ledcAttachChannel(PIN_PAN, SERVO_FREQ, SERVO_RES, CH_PAN);
  ledcAttachChannel(PIN_TILT, SERVO_FREQ, SERVO_RES, CH_TILT);
  centerServos();

  // ---- 新增：V19 小车硬件（电机 / 循迹传感器 / PS2 手柄）----
  digitalWrite(PIN_STBY, LOW);              // 先把 STBY 拉低再开输出，避免上电乱转
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, LOW);
  pinMode(PIN_AIN1_L, OUTPUT); pinMode(PIN_AIN2_L, OUTPUT);
  pinMode(PIN_AIN1_R, OUTPUT); pinMode(PIN_AIN2_R, OUTPUT);
  // 电机固定占 LEDC 2/3：舵机在上面已占 0/1，group0 内不同定时器，互不影响
  const bool leftPwmOk  = ledcAttachChannel(PIN_PWM_L, TB_PWM_FREQ, TB_PWM_RES, CH_MOTOR_L);
  const bool rightPwmOk = ledcAttachChannel(PIN_PWM_R, TB_PWM_FREQ, TB_PWM_RES, CH_MOTOR_R);
  motorPwmReady = leftPwmOk && rightPwmOk;
  if (!motorPwmReady) Serial.println("MOTOR FAULT: 电机 PWM 初始化失败，不能 arm");
  tbHardStop();                            // IN 拉低 / PWM=0

  lineBusReady = Wire.begin(I2C_SDA, I2C_SCL, I2C_FREQ);   // 8 路循迹传感器
  Wire.setTimeOut(1000);                   // 20us 太短会 I2C 超时，改 1000
  if (!lineBusReady) Serial.println("LINE FAULT: I2C 初始化失败");

  ps2.begin();
  lastPs2Poll = lastValidPs2 = millis();
  lastGimbalSend = millis();

  Serial.println("[V19] 小车模块已并入: TB6612 + PS2 + 8 路循迹 (云台仍走 K230 $KT)");
  Serial.print("LEDC 通道: PAN="); Serial.print(CH_PAN);
  Serial.print(" TILT="); Serial.print(CH_TILT);
  Serial.print(" MOTOR_L="); Serial.print(CH_MOTOR_L);
  Serial.print("(ok="); Serial.print(leftPwmOk);
  Serial.print(") MOTOR_R="); Serial.print(CH_MOTOR_R);
  Serial.print("(ok="); Serial.print(rightPwmOk); Serial.println(")");
  Serial.println("TB6612 LEFT: PWM33/IN1_2/IN2_4  RIGHT: PWM32/IN1_13/IN2_26  STBY=GPIO25");
  Serial.print("PS2 pins: DAT="); Serial.print(PS2_DAT);
  Serial.print(" CMD="); Serial.print(PS2_CMD);
  Serial.print(" ATT="); Serial.print(PS2_ATT);
  Serial.print(" CLK="); Serial.println(PS2_CLK);
  Serial.println("上电: 无模式 (按手柄 R2 才进跟踪 / 才让 K230 跑识别); 小车无模式");
  Serial.println("手柄: L1=小车手动 L2=循迹 R1=云台手动 R2=云台跟踪; 电机需 arm");

  Serial.println();
  Serial.println("ESP32_Gimbal_K230 ready");
  Serial.println("UART2: RX=16 TX=17 115200 | PAN=27 TILT=14");
  help();
  // 上电默认无模式：电机不使能、云台保持中位；由手柄 L1/L2/R1/R2 选模式
  setCarMode(BOOT_CAR_MODE);
  setGimbalMode(BOOT_GIMBAL_MODE);
  K230.println("$E,READY,ESP32_GIMBAL_K230!");
}

void loop() {
  static char kbuf[128];
  static size_t kused = 0;
  static char ubuf[128];
  static size_t uused = 0;
  static uint32_t loops = 0, lastLoops = 0;

  ++loops;
  // 新增：先分发，$KT 帧原样交给 handleTargetLine；DONE 用于巡线继续
  readLineFrom(K230, kbuf, sizeof(kbuf), kused, onK230Line, false);
  readLineFrom(Serial, ubuf, sizeof(ubuf), uused, onUsbLine, true);

  const uint32_t now = millis();
  // ---- 新增：手柄 / 循迹 / 电机服务（云台跟踪仍由下面的 updateTrack 负责）----
  padService(now);
  padButtons(now);
  lineService(now);
  motorTestService(now);
  motorService(now);
  carStatusLine(now);
  linkDiag(now);

  updateTrack(now);

  static uint32_t lastPrint = 0;
  static int lastPanForRate = SERVO_MID_US;
  if (now - lastPrint >= 1000) {
    const uint32_t loopHz = loops - lastLoops;
    lastLoops = loops;
    const int panRate = panUs - lastPanForRate;
    lastPanForRate = panUs;
    lastPrint = now;
    Serial.print("[GIMBAL] pan="); Serial.print(panUs);
    Serial.print(" tilt="); Serial.print(tiltUs);
    Serial.print(" state="); Serial.print(trackState);
    Serial.print(" frames="); Serial.print(frames);
    Serial.print(" bad="); Serial.print(badFrames);
    Serial.print(" hz="); Serial.print(loopHz);
    Serial.print(" panrate="); Serial.print(panRate);
    Serial.print(" err=");
    if (lastFrameMs) {
      const int ax = (lastBx >= 0) ? lastBx : lastCx;
      const int ay = (lastBy >= 0) ? lastBy : lastCy;
      Serial.print(ax - CENTER_X); Serial.print(","); Serial.print(ay - CENTER_Y);
    } else {
      Serial.print("-");
    }
    Serial.println();
  }
}
