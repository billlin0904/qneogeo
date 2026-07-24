# qneogeo

qneogeo 是一個使用 Qt 6 製作的 Neo Geo 前端。模擬核心目前可使用
`neocd_libretro` 執行 Neo Geo CD，並保留 FBNeo libretro core 的前端切換入口。
畫面由 OpenGL 顯示，音訊輸出使用 RtAudio，並支援 XInput、鍵盤按鍵配置、
存取狀態、FPS 顯示與多種影像濾鏡。

> 本專案不包含 BIOS、遊戲映像或任何商業 ROM/CD 內容。請使用你合法擁有的
> Neo Geo CD BIOS 與遊戲備份。

## Features

- 內建 `neocd_libretro` CMake/Visual Studio build target
- `File > Core` 可切換 Neo Geo CD / FBNeo Neo Geo Arcade frontend mode
- Qt 6 視窗前端與 OpenGL 畫面輸出
- Neo Geo CD `.cue` / `.chd` 與 FBNeo `.zip` / `.7z` 遊戲掃描清單
- 鍵盤與 XInput 控制器配置
- `Arcade SOCD Clean` 與 `Motion Assist` 方向輸入輔助
- WASAPI/RtAudio shared audio output
- libsamplerate 重採樣到 48 kHz
- 暫停、重新模擬、視窗失焦自動暫停、save state / load state
- FPS overlay
- 影像濾鏡：
  - Nearest
  - Linear
  - xBRZ Freescale
  - libretro xBRZ Freescale
  - libretro 6xBRZ
  - zfast-CRT
  - zfast-LCD
  - Scanline-fract
- Release build 使用 Windows GUI subsystem，不會開啟 console 視窗

## Requirements

- Windows 10/11 x64
- Visual Studio 2026 generator
- Qt 6.8.3 MSVC x64，預設路徑：

```text
C:\Qt\6.8.3\msvc2022_64
```

- vcpkg，預設路徑：

```text
F:\vcpkg\vcpkg
```

目前 CMake preset 會使用：

```text
F:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake
```

需要的 vcpkg package 會由 CMake 尋找：

- zlib
- zstd
- ogg
- vorbis
- rtaudio
- libsamplerate

vcpkg 目前沒有 `fbneo_libretro` port。`QNEOGEO_FETCH_FBNEO_CORE` 預設開啟，
CMake 會從 libretro buildbot 下載 Windows x64 `fbneo_libretro.dll` 到本機
`downloads/` 快取，build 後複製到 exe 旁邊。

## Repository Setup

第一次 clone 後請初始化 submodule：

```powershell
git submodule update --init --recursive
```

`neocd_libretro` 位於：

```text
thirdparty/neocd_libretro
```

## Build

Configure：

```powershell
cmake --preset vs2026-x64
```

Debug：

```powershell
cmake --build --preset vs2026-x64-debug --target qneogeo
```

Release：

```powershell
cmake --build --preset vs2026-x64-release --target qneogeo
```

輸出位置：

```text
build-vs2026-x64\Debug\qneogeo.exe
build-vs2026-x64\Release\qneogeo.exe
```

## Folder Layout

執行時會使用專案根目錄下的資料夾：

```text
system/        Neo Geo CD BIOS
roms/neocd/    Neo Geo CD games
roms/fbneo/    FBNeo arcade ROM sets
saves/         save data and save states
config/        input and video settings
shaders/       GLSL shader files
```

這些本機資料夾預設不簽入 git。

## BIOS

請把 Neo Geo CD BIOS 放在：

```text
system/
```

常見 BIOS 檔名依 `neocd_libretro` 支援狀況而定。若 BIOS 不正確，遊戲可能只會
停在 Neo Geo CD player 或 BIOS 畫面。

## Games

使用：

```text
File > Core
```

可以切換目前前端 core mode。選擇會記錄到：

```text
config/input.ini
```

Region 與 Neo Geo mode 可由選單設定：

```text
File > Region
File > Mode
File > CPU Clock
```

`Region` 會套用到 NeoCD 與 FBNeo。`Mode` 目前保留 MVS (Arcade) 與
AES (Console)，並會在 FBNeo 內部轉成 `fbneo-neogeo-mode`，例如 MVS Japan
或 AES Japan。遊戲載入中切換這些選項時，前端會重新載入目前遊戲讓 BIOS
設定生效。

`CPU Clock` 作用於 FBNeo 的 `fbneo-cpu-speed-adjust`，可選 50%、100%、150%、200%，用來
調整部分遊戲的原生 slowdown 行為。

Neo Geo CD 遊戲放在：

```text
roms/neocd/
```

目前清單會掃描：

- `.cue`
- `.chd`

`.cue` 遊戲建議維持獨立資料夾，例如：

```text
roms/neocd/
  King of Fighters '98/
    King of Fighters '98.cue
    track01.bin
    track02.wav
```

FBNeo arcade ROM sets 放在：

```text
roms/fbneo/
```

目前清單會掃描：

- `.zip`
- `.7z`

FBNeo core DLL 需放在 exe 旁邊。預設 CMake 會自動下載並複製：

```text
build-vs2026-x64\Debug\fbneo_libretro.dll
build-vs2026-x64\Release\fbneo_libretro.dll
```

下載來源：

```text
https://buildbot.libretro.com/nightly/windows/x86_64/latest/fbneo_libretro.dll.zip
```

啟動後使用：

```text
File > Load Game
```

## Controls

使用：

```text
File > Configuration Input
```

可以設定鍵盤與 XInput 對應。設定會寫入：

```text
config/input.ini
```

Configuration Input dialog 提供：

- `Arcade SOCD Clean`
  - 清理鍵盤不可能出現在街機搖桿上的相反方向，例如 `Left + Right`
    或 `Up + Down`
- `Motion Assist`
  - 斜方向快速跳轉時補一幀中間方向，例如 `↙ -> ↓ -> ↘`

這兩個選項會套用到鍵盤與 XInput 的方向輸入，不影響攻擊按鈕。

## Video Filters

使用：

```text
Video > Filter
```

Filter 選擇會記錄到：

```text
config/input.ini
```

目前主畫面 shader 由下列檔案載入：

```text
shaders/frame.vert
shaders/frame.frag
```

libretro shader 參考檔：

```text
shaders/xbrz-freescale.glsl
shaders/6xbrz.glsl
shaders/zfast_crt.glsl
shaders/zfast_lcd.glsl
shaders/scanline-fract.glsl
```

Build 後 `shaders/` 會自動複製到 exe 旁邊。

## Save States

使用：

```text
File > Save State > Slot 1 ... Slot 10
File > Load State > Slot 1 ... Slot 10
```

Save state 會放在：

```text
saves/states/
```

每個遊戲最多可保存 10 個 slot，檔名格式為：

```text
<game>.slot1.state ... <game>.slot10.state
```

重新模擬目前載入的遊戲：

```text
File > Reset Emulation
```

## Memory View

使用：

```text
Tools > Memory View
```

可以查看 libretro core 暴露的 `RETRO_MEMORY_SYSTEM_RAM`。Address 可輸入
Neo Geo 68K RAM 位址，例如：

```text
0x100000
0x108118
10FD94
```

表格左側顯示 `0x100000 + RAM offset` 的 68K address，右側顯示連續記憶體值。
在表格上按右鍵可切換 `1 Byte` 或 `2 Bytes` 顯示。

## Notes

- Debug build 保留 console，方便查看 shader、libretro 與音訊 log。
- Release build 不會開 console。
- `config/`、`roms/`、`saves/`、`system/` 是本機資料，不會進入 git。

## KOF98 PPO Training

`fbneo_training.dll` 提供無音訊的 FBNeo 訓練 runtime。Python 透過 ctypes 使用
29 個高階 Action，C++ 負責角色朝向、逐 frame 輸入腳本與派生技 queue。

訓練 state：

```text
saves/states/kof98.slot1.state  Combo profile
saves/states/kof98.slot2.state  Fight profile
saves/states/kof98.slot3.state  零氣的琴月陽 Combo scenarios
```

Combo mask curriculum：

```text
strict    Idle + 正確 phase Action
guided    strict + 兩個物理合法干擾 Action
physical  空閒時開放全部 Action，忙碌時只允許 Idle 與合法 queued follow-up
```

目前 Combo scenarios 可用 `--combo-suite kyo29` 一次載入，也可用
`--combo-scenario NAME=STATE` 重複指定。內容包含毒咬三段、
前 B 接琴月陽／大蛇薙／R.E.D. Kick／荒咬，以及
`Close C -> 七十五式改 -> 大蛇薙／R.E.D. Kick／琴月陽／荒咬`，以及角落限定的
`Close C -> 七十五式改 -> 荒咬 -> 八錆 -> 砌穿`，也包含
`蹲 B -> 蹲 A -> 無式` 的 7 Hit 低段確認連段。

修改 Action frame 後先執行 deterministic 驗證：

```powershell
C:\Users\User\anaconda3\envs\KofAI\python.exe tools\verify_kof98_actions.py
```

StrategyV3-A action set 使用 version 2：P1 的 Action 1～4 是剛好 4 frame
的移動／防禦 chunk，不再附加 2 frame neutral；repeat4 的下一次決策可無縫
延長同方向、選 Idle 放開，或立即改成攻擊。P2 仍使用原本 4+2 腳本，避免
同時改變訓練對手。DLL 會回報 API version、public action count、action-set
version 與 chunk frames，run manifest 和新模型也會保存相同 metadata。
舊 checkpoint 若要在新 action semantics 下 fine-tune，必須明確加入
`--allow-action-set-migration`；普通 `--resume` 會拒絕靜默混用。

StrategyV3-A2 使用 deterministic Level Recipe 建立 reverse curriculum。
Recipe 不會在招式播放途中另存 state；它保存安全的 neutral state、P2 動作、
P2 開始延遲／prelude 幀數與 Oracle 動作，reset 時重新播放相同情境。這避免
libretro state 沒有包含 C++ action queue 與 Python phase 的半套快照問題。
先執行 Oracle scanner 找出實際可成功的幀窗口：

```powershell
C:\Users\User\anaconda3\envs\KofAI\python.exe tools\kof98_oracle_scanner.py
```

結果寫入 `ai_logs/oracle/kof98_v3a2_oracle.json`，可訓練的關卡寫入
`ai_logs/oracle/kof98_v3a2_recipes.json`。Base state 必須通過 DLL 的
`snapshot_safe`：回合進行中、雙方存活且有座標、雙方可行動、P1/P2 input
script 與 queue 全空。Approach 成功也只認 P1 Action 1 實際朝對手移動至少
12 px；P2 自己靠近不再冒領。

Oracle 產生 recipe bank 後，可在 VS Code 選
`Train StrategyV3-B (Oracle Curriculum)`。兩個 recipe worker 每次 reset 輪替
防禦、對空、接近與 hit-confirm 關卡；其餘 Fight worker 保持一般實戰。
Recipe 只簡化起始狀態，Action Mask 永遠使用完整 Physical legality，不再把
正確招式藏在 Targeted Mask 裡。`symmetric_tactical_rm_v3` 只在真實事件成立時
給課程 reward：成功 `+3`、事件失敗 `-0.5`、超時 `-0.25`；普通 Physical
Fight 仍維持對稱 HP＋勝負 reward。

V3-B 另外使用小權重的 Oracle imitation loss，把 recipe 的成功輸入當成輔助
標籤；PPO 每一步仍可從完整 Physical Mask 自由選擇，Oracle 不會縮小合法 Action
集合。輔助權重會在 `--oracle-teacher-decay-steps` 內線性降為 0，讓最終策略依靠
遊戲 observation 自己決策。這可教回舊模型幾乎不探索的前進、防禦與鬼燒對空，
同時保留 PPO 的傷害與勝負目標。

主要 TensorBoard 指標：

```text
kof_level/<recipe>/success_rate
kof_level/<recipe>/failure_rate
kof_level/<recipe>/timeout_rate
kof_fight_targeted/<task>/reward_machine_success_rate
kof_fight_physical/*
oracle_teacher/accuracy
oracle_teacher/loss
oracle_teacher/weight
oracle_teacher/replay_size
oracle_teacher/labels_collected
oracle_teacher/<task>_labels_collected
```

`oracle_teacher/*` 只表示模型是否能從完整 mask 中重現 Oracle Action；
`kof_level/*` 是上課成績。是否真正轉移仍要看 `kof_fight_physical` 的防禦、
對空、安全接近與 hit-confirm 條件成功率，不能只看 imitation accuracy 或
recipe success。完整整合驗證會實際載入 FBNeo 與所有 recipe：

```powershell
C:\Users\User\anaconda3\envs\KofAI\python.exe tools\verify_kof98_actions.py
```

Action space 固定為 `Discrete(29)`。StrategyV2 將 observation 從 26 擴為 140，
新增雙方速度、角色 raw status、可行動狀態、畫面邊界、目前／排隊 Action、
Action 經過與剩餘 frame，以及 combat phase。它不猜未驗證的 hitstun/blockstun。

VS Code 的 `Train StrategyV2 (Migrate + Kickstart)` 會把現有 26 維模型的第一層
權重複製到新模型，新欄位權重初始化為 0；遷移當下的 action probabilities 與
value 保持不變。舊模型同時作為會自動退火的 teacher，避免一開始忘記既有連段。
既有 StrategyV2 模型完成後，使用 `Train StrategyV2 (Tactics Fine-tune)`：2 個
Combo env 會在 reset 時輪替全部 11 套連段；2 個 Targeted Fight env 輪替防禦、
對空、接近、hit confirm；6 個 Physical Fight env 輪替四種 P2 style。樣本比例仍是
20%／20%／60%，但只需同時載入 10 個 FBNeo，避免 20 個程序耗盡記憶體。這個設定
不載入舊 teacher model，完整 Fight reward 仍只使用對稱 HP 差與勝負。
TensorBoard 會把 Targeted 與 Physical 分開，並記錄只以自由決策為分母的 Action
占比，以及「壓力下成功防禦／空中對手被鬼燒命中／安全進入有效距離／起手確認後
接續命中」四種條件成功率；因此老師環境會按指定招，不會被誤算成實戰策略轉移。

StrategyV2 的完整 Fight reward 只使用對稱 HP 差與勝負。距離、hitbox overlap、
招式名稱、快速勝利及長連段 bonus 不再混入完整對戰；指定戰術的 shaping 留在
Combo／Fight curriculum，避免對單一 scripted bot 學出刷分捷徑。

### 在 qneogeo 與 PPO 對戰

切換到 `FBNeo Training DLL` core、載入 KOF98，再勾選
`Tools > GamePlay With AI (P2)`。正式驗收時保持
`Tools > P2 AI Pure Policy` 勾選；此模式只載入 Fight model，不切換 Combo model，
也不把遠距蹲 A／蹲 B 改寫成前進。取消勾選才會啟用原本較偏展示用途的雙模型
切換與遠距 Action override。qneogeo 會讓常駐 Python process 載入：

```text
trained_models/kof98_strategy_v3b_oracle_curriculum_ppo_final.zip
```

Qt 每 4 個模擬 frame 非同步送出 P2 視角的 26 或 140 維 observation 與 physical
action mask；Python 只做 MaskablePPO 推論，P2 的逐 frame 指令與派生技 queue 仍由
`fbneo_training.dll` 執行。關閉選項後，P2 鍵盤輸入會恢復。

Fight 模型可使用 StrategyV2，而 Combo 模型仍可保留 26 維；推論 bridge 會在
切換到 Combo policy 時自動取 observation 前 26 欄。

預設 Python 為：

```text
C:/Users/User/anaconda3/envs/KofAI/python.exe
```

可在 `config/input.ini` 覆寫：

```ini
[AI]
PythonExecutable=C:/path/to/python.exe
P2Model=F:/path/to/model.zip
P2ComboModel=F:/path/to/combo_model.zip
```
