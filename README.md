# qneogeo

qneogeo 是一個使用 Qt 6 製作的 Neo Geo CD 前端。模擬核心使用
`neocd_libretro`，畫面由 OpenGL 顯示，音訊輸出使用 RtAudio，並支援
XInput、鍵盤按鍵配置、存取狀態、FPS 顯示與多種影像濾鏡。

> 本專案不包含 BIOS、遊戲映像或任何商業 ROM/CD 內容。請使用你合法擁有的
> Neo Geo CD BIOS 與遊戲備份。

## Features

- 內建 `neocd_libretro` CMake/Visual Studio build target
- Qt 6 視窗前端與 OpenGL 畫面輸出
- Neo Geo CD `.cue` / `.chd` 遊戲掃描清單
- 鍵盤與 XInput 控制器配置
- `Arcade SOCD Clean` 與 `Keyboard Motion Assist` 鍵盤出招輔助
- WASAPI/RtAudio shared audio output
- libsamplerate 重採樣到 48 kHz
- 暫停、視窗失焦自動暫停、save state / load state
- FPS overlay
- 影像濾鏡：
  - Nearest
  - Linear
  - Super2xSaI
  - xBRZ Freescale
  - libretro xBRZ Freescale
  - libretro 6xBRZ
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

請把 Neo Geo CD 遊戲放在：

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

啟動後使用：

```text
Menu > Load Game
```

## Controls

使用：

```text
Menu > Configuration Input
```

可以設定鍵盤與 XInput 對應。設定會寫入：

```text
config/input.ini
```

Input menu 提供：

- `Arcade SOCD Clean`
  - 清理鍵盤不可能出現在街機搖桿上的相反方向，例如 `Left + Right`
    或 `Up + Down`
- `Keyboard Motion Assist`
  - 鍵盤斜方向快速跳轉時補一幀中間方向，例如 `↙ -> ↓ -> ↘`

這兩個選項只作用於鍵盤，不影響 XInput。

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
```

Build 後 `shaders/` 會自動複製到 exe 旁邊。

## Save States

使用：

```text
State > Save State
State > Load State
```

Save state 會放在：

```text
saves/states/
```

## Notes

- Debug build 保留 console，方便查看 shader、libretro 與音訊 log。
- Release build 不會開 console。
- `config/`、`roms/`、`saves/`、`system/` 是本機資料，不會進入 git。
