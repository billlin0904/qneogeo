# KOF98 反應硬直與可行動幀驗證報告

產生日期：2026-07-23
專案：`F:\Source\qneogeo`
用途：提供另一個模型或研究者獨立比對 KOF98 的 Hitstop、Hitstun、
Blockstun、D2:D3、E3 與可行動幀判定。

## 1. 測試環境

| 項目 | 內容 |
| --- | --- |
| 遊戲 | 原版 Neo Geo `kof98` |
| Emulator core | FBNeo libretro |
| Core log version | `v1.0.0.03 260703 GITc7b89b7` |
| Runtime DLL | `build-vs2026-x64/Release/fbneo_training.dll` |
| Core DLL | `downloads/fbneo_libretro/fbneo_libretro.dll` |
| ROM | `roms/fbneo/kof98.zip` |
| Savestate | `saves/states/kof98.slot1.state` |
| 角色 | P1/P2 均為京 |
| 驗證器 | `tools/verify_kof98_actions.py` |

測試直接使用 `kof_env_set_joypad_for_port()` 送入 raw joypad 狀態，
每次只執行一個 emulator frame，再讀取 FBNeo system RAM。這些測試不依賴
Qt 畫面更新頻率，也不依賴 PPO 的 `action_repeat`。

## 2. RAM 讀取規則

FBNeo 暴露的 68K RAM 使用 word-swapped byte layout。讀取一個邏輯 byte：

```cpp
logical_byte = system_ram[address ^ 1];
```

`D2:D3` 以 Motorola 68000 big-endian signed 16-bit 組合：

```cpp
uint16_t raw =
    (static_cast<uint16_t>(d2) << 8) |
    static_cast<uint16_t>(d3);

int16_t reaction_counter =
    static_cast<int16_t>(raw);
```

例：

```text
FF:FF = -1
FF:FE = -2
00:10 = 16
00:08 = 8
```

玩家結構基址：

```text
P1 base = 0x108100，system RAM offset = 0x8100
P2 base = 0x108300，system RAM offset = 0x8300
```

本報告涉及的欄位：

| 欄位 | P1 | P2 | 用途 |
| --- | ---: | ---: | --- |
| D2:D3 | `0x1081D2` | `0x1083D2` | 反應子階段 counter |
| E0 | `0x1081E0` | `0x1083E0` | 角色動作狀態觀察值 |
| E3 | `0x1081E3` | `0x1083E3` | 接觸／反應 bitfield |
| Guard Crush | `0x108247` | `0x108447` | 防禦耐久值 |

## 3. E3 與 D2:D3 的工作假說

目前驗證資料支持：

```cpp
const uint8_t reaction = e3 & 0x60;

const bool guard_signature = reaction == 0x20;
const bool hit_signature   = reaction == 0x60;
```

但 E3 只是分類特徵，不足以單獨證明 Block 或 Hit：

```text
E3 & 0x60 == 0x20：已觀察到地面防禦反應
E3 & 0x60 == 0x60：已觀察到命中反應，也會出現在 Guard Crush
```

`D2:D3` 不是通用的 `blockstun_remaining`。它會被地面防禦、地面受擊、
空中受擊與擊倒流程重用。只有在已經由接觸證據確認是地面防禦後，
`0..N` 才能暫時解釋為防禦反應倒數。

## 4. 地面普通技驗證

### 4.1 自動測試結果

| 測試 | Hitstop | Hit/Block 反應 | D2:D3 倒數 | 總反應 |
| --- | ---: | ---: | --- | ---: |
| 地面輕攻擊命中 | 11F | 11F | `8→0` 前另有 2F 命中延遲 | 22F |
| 地面重攻擊命中 P1→P2 | 11F | 19F | `16→0` 前另有 2F 命中延遲 | 30F |
| 地面重攻擊命中 P2→P1 | 11F | 19F | `16→0` 前另有 2F 命中延遲 | 30F |
| 地面輕攻擊防禦 | 11F | 9F | `8,7,...,0` | 20F |
| 地面重攻擊防禦 | 11F | 17F | `16,15,...,0` | 28F |

自動斷言：

```text
light-hit-11+11             PASS
p1-heavy-hit-11+19         PASS
p2-heavy-hit-11+19         PASS
light-block-11+9           PASS
heavy-block-11+17          PASS
light-d2d3-8-to-0          PASS
heavy-d2d3-16-to-0         PASS
```

Steam KOF98UMFE 結構中曾被標成 hitstun 的 `D4:D5`，在上述原版
KOF98 測試期間皆為 0，因此不能直接套用到本版本：

```text
steam-d4-rejected          PASS
```

## 5. 防禦解除與第一個可輸入幀

### 5.1 判定條件

Raw RAM 中已驗證的地面防禦結束候選是：

```cpp
const uint8_t previous_kind = previous_e3 & 0x60;
const uint8_t current_kind = current_e3 & 0x60;

const bool raw_reaction_end_candidate =
    previous_counter == 0 &&
    current_counter == -1 &&
    previous_kind == 0x20 &&
    current_kind == 0x00;
```

`current_kind` 必須回到 `0x00`。若同幀轉成 `0x60`，代表進入另一個命中
反應，不得把它當成 Blockstun End 或開啟確反窗口。

Raw 候選不能直接產生事件，必須綁定既有的 Guard tracker：

```cpp
const bool blockstun_ended =
    tracker.active &&
    tracker.confirmed_block_contact &&
    tracker.countdown_loaded &&
    !tracker.refresh_pending &&
    !tracker.end_emitted &&
    raw_reaction_end_candidate;
```

`0→-2` 或其他未知負值不視為防禦結束。未知狀態採保守失敗，只記錄
diagnostic，不產生確反 reward。

建議的最小 tracker 資料：

```cpp
enum class GuardTrackerPhase {
    Idle,
    WaitingForCountdown,
    CountdownActive,
    RefreshPending,
};

struct GuardReactionTracker {
    GuardTrackerPhase phase = GuardTrackerPhase::Idle;
    bool active = false;
    bool confirmed_block_contact = false;
    bool countdown_loaded = false;
    bool refresh_pending = false;
    bool start_emitted = false;
    bool end_emitted = false;
    bool manual_success_emitted = false;

    uint64_t contact_frame = 0;
    uint64_t not_before_frame = 0;
    uint64_t timeout_frame = 0;
    uint64_t candidate_counter_frame = 0;
    uint64_t countdown_start_frame = 0;

    int16_t candidate_counter = -1;
    int32_t last_contact_action_id = -1;
    uint32_t last_contact_action_serial = 0;
};
```

#### Stale counter 防護

完整對戰 trace 證明接觸當幀的 `D2:D3` 可能殘留 `503`、`11` 或 `0`。
每次已分類的 Block Contact 必須先建立：

```cpp
tracker.contact_frame = current_engine_frame;
tracker.not_before_frame =
    tracker.contact_frame +
    std::max<int32_t>(1, hit_guard_stop_raw);
tracker.timeout_frame = tracker.contact_frame + 240;
tracker.candidate_counter = -1;
tracker.candidate_counter_frame = 0;
```

在 `not_before_frame` 之前，不得把任何 `D2:D3` 正值當成新倒數。到達
`not_before_frame` 後，仍必須觀察連續兩幀的 `N→N-1`，才確認倒數已載入：

```cpp
if (current_engine_frame >= tracker.not_before_frame &&
    current_kind == 0x20 &&
    current_counter > 0 &&
    tracker.candidate_counter < 0) {
    tracker.candidate_counter = current_counter;
    tracker.candidate_counter_frame = current_engine_frame;
}

if (tracker.candidate_counter > 0) {
    if (current_kind != 0x20 || current_counter < 0) {
        abort_tracker_with_diagnostic();
    } else if (current_counter == tracker.candidate_counter) {
        // Hitstop／refresh freeze；保留 candidate 繼續等待。
    } else if (current_counter == tracker.candidate_counter - 1) {
        tracker.countdown_loaded = true;
        tracker.countdown_start_frame =
            tracker.candidate_counter_frame;

        tracker.refresh_pending = false;
        tracker.phase = GuardTrackerPhase::CountdownActive;

        tracker.candidate_counter = -1;
        tracker.candidate_counter_frame = 0;

        if (!tracker.start_emitted) {
            record_internal_blockstun_start(
                tracker.countdown_start_frame);
            tracker.start_emitted = true;
        } else {
            record_counter_reload_telemetry();
        }
    } else if (current_counter > 0) {
        // 載入值改變但尚未形成 N→N-1，從目前值重新建立 candidate。
        tracker.candidate_counter = current_counter;
        tracker.candidate_counter_frame =
            current_engine_frame;
    }
}
```

第一幀看到 `N` 時只保存 candidate；下一幀真的變成 `N-1` 才確認。事件的
邏輯 start frame 是 candidate 所在幀。`N→N` 表示 freeze，繼續等待；
`N→其他正數` 重新建立 candidate；負值或 E3 異常則保守中止。

#### 多段防禦 Refresh

同一 defender 同一時間只維持一個 Guard tracker。Tracker 尚未結束時收到
新的 Block Contact：

1. 不建立第二個 tracker。
2. 更新最後接觸的 `action_id`、`action_serial` 與 contact evidence。
3. 重新計算 `not_before_frame` 與 `timeout_frame`。
4. 設定 `refresh_pending=true`，等待新的 `N→N-1`。
5. Counter reload 只記 telemetry，不重複發送 `BLOCKSTUN_STARTED`。
6. 僅在最後一次 refresh 完成後的 `0→-1` 發送一次 `BLOCKSTUN_ENDED`。

確認 refresh 的新 `N→N-1` 時，必須將：

```cpp
tracker.refresh_pending = false;
tracker.phase = GuardTrackerPhase::CountdownActive;
```

Refresh 不得清除 `start_emitted` 或 `manual_success_emitted`，因此同一條
guard string 最多只產生一次 START 與一次 `MANUAL_BLOCK_SUCCESS`。

`BLOCKSTUN_STARTED` 預設只作為 C++ tracker 內部狀態與 telemetry，不要求
回填既有 Python Step Events ABI。若未來要讓 Python 收到 START，事件必須
新增 absolute logical engine frame；不能只用目前 step 內的相對位置回填
上一個 `runFrames()` 批次。

狀態流程：

```text
Idle
  → Contact Classified
  → Waiting For Countdown
  → Countdown Active
      ├─ New Block Contact
      │    → Refresh Pending
      │    → Countdown Active
      └─ 0→-1 and E3 0x20→0x00
           → Reaction Ended
           → Idle
```

超過最後一次接觸 240F、遇到未知負值、E3 非預期消失或轉為 Hit reaction
時，清除 tracker 並記錄原因，不發 END 或 Reward。

每幀處理順序必須先檢查合法 END，再處理「E3 非預期消失」：

```text
1. 處理 explicit／in-step reset boundary
2. 分類本幀的新 Contact／Refresh
3. 檢查已綁定 tracker 的合法 0→-1、0x20→0x00 END
4. 若尚未 END，再把其他 E3 消失／轉換視為 abort diagnostic
5. 更新 previous-frame baseline
```

否則合法 END 本身的 `0x20→0x00` 會先被 abort 路徑吃掉。

### 5.2 Reset Protocol

以下狀況必須使 reaction baseline 失效：

```text
runtime reset
load state / unserialize
round 或角色切換
HP 非戰鬥性回升
game phase 切換
角色座標失效
```

API 呼叫前發生的 explicit reset／load state 尚未執行本批 emulator frames，
可以清除整個事件批次：

```text
previous-frame RAM baseline
Guard／Hit reaction trackers
pending action_id／action_serial attribution
counter／punish reward windows
Step Event V1/V2/V3 buffers 與 dropped counters
input history
```

若在一次 `runFrames(N)` 途中偵測 round／角色切換或座標失效，不能清除本批
較早 frame 已寫入的合法事件。例如第 1F 是 KO 命中、第 3F 才切換回合，
第 1F 的命中必須保留。

In-step transition 應：

```text
保留 boundary 之前已寫入的 Step Events 與 dropped count
增加 event_epoch
可選擇附加 RESET_BOUNDARY telemetry
清除 reaction tracker、pending attribution、reward window 與 input history
使 RAM baseline 失效
抑制 boundary 之後的事件，直到 baseline 重建完成
```

每筆 Step Event 應帶 `event_epoch`；consumer 不得把不同 epoch 的 contact、
action serial 或 counter window 串在一起。

Reset 後第一個有效 frame 只用來重建 baseline，不比較 HP、Guard Crush、
Combo 或 reaction delta，也不產生接觸事件。至少取得下一個連續有效 frame
後才恢復分類。

### 5.3 Raw joypad 對稱測試

P1 與 P2 各跑 10 次，共測三種輸入：

1. 每隔一幀重新按 A，且按鍵邊緣落在反應結束幀。
2. 按鍵邊緣落在 `counter=0` 幀，下一幀再次按 A。
3. 從硬直中持續壓住 A，不重新產生按鍵邊緣。

| 驗證項目 | P1 | P2 |
| --- | ---: | ---: |
| `0→-1` 結束當幀接受新 A 邊緣 | 10/10 | 10/10 |
| `counter=0` 幀的新 A 尚未成立 | 10/10 | 10/10 |
| 下一幀重新按 A 可以成立 | 10/10 | 10/10 |
| 持續壓住 A 不會在解除後自動出招 | 10/10 | 10/10 |

這表示可行動邊界是：

```text
counter = 0：仍在反應硬直
counter: 0 → -1 且 Guard E3 清除：第一個可接受新按鍵邊緣的幀
```

### 5.4 人工 CSV 交叉驗證

來源：

```text
logs/ram_traces/kof98_20260723_134130.csv
```

P2 防禦遠 C：

```text
F647：接觸，P2 E3=0x20，Guard Crush spent=9
F658：D2:D3 載入 16
F658～F674：16→0
F675：0→-1，E3 0x20→0
F676：人工重新按 A
```

這份人工 CSV 的 A 是在反應結束後一幀才重新按下；精確的「結束當幀」
邊界由上述 raw joypad 自動測試補足。

### 5.5 Engine Open Frame 與 Policy Visible Frame

Raw joypad 測試得到的是：

```text
engine_open_frame
```

也就是 emulator 在 `0→-1` 當幀可接受新的按鍵邊緣。PPO 使用固定
`action_repeat` 時，並不一定能在同一幀看到事件並決策：

```text
policy_visible_frame
= 包含 engine_open_frame 的 kof_env_step 返回後，
  下一次 action 能套用到 emulator 的第一幀
```

例如一次 step 執行 4 個 emulator frames，若 open event 發生在該 step
第一幀，policy 最晚要到下一個 4-frame chunk 才能反應。確反窗口與事件
timeout 必須使用絕對 `engine_frame`；TensorBoard 另外記錄：

```text
engine_open_frame
policy_visible_frame
policy_observation_delay_frames
```

不能把 raw 測試的 engine open frame 直接描述成 PPO 的可決策幀。

## 6. 低跳、正常跳與 CD

### 6.1 跳躍分類方式

直接控制 P2 raw `UP+FORWARD` 的持續時間：

```text
按上 2F：低跳，最低 Y 約 155，接觸時 E0=0x0B
按上 6F：正常跳，最低 Y 約 121，接觸時 E0=0x03
```

接觸時 P2 Y 均小於 185，確認不是落地後的站立攻擊。

### 6.2 自動測試結果

| 動作 | 接觸時 Y | 最低 Y | D2:D3 | 反應長度 |
| --- | ---: | ---: | --- | ---: |
| 低跳 C | 155 | 155 | `8→0` | 9F |
| 低跳 D | 157 | 155 | `8→0` | 9F |
| 正常跳 C | 153 | 121 | `16→0` | 17F |
| 正常跳 D | 159 | 121 | `16→0` | 17F |
| 地面 CD 防禦 | 200 | 200 | `20→0` | 21F |

地面 CD 未防禦命中時：

```text
D2:D3 先出現 4→0，之後進入 -2 的擊倒流程
只產生 Clean Hit 類事件
沒有產生 BLOCKSTUN_STARTED / BLOCKSTUN_ENDED
```

這是 `D2:D3` 不能被全域當成 Blockstun timer 的直接反例。

### 6.3 與日文硬直資料的比對

使用者提供的《硬直時間》明確列出：

```text
地上技被地上防禦：
弱攻擊       9F
強攻擊      17F
ふっとばし  21F

空中技被地上防禦：
小跳攻擊     9F
弱攻擊       9F
強攻擊      17F
ふっとばし  21F
```

同一份資料也列出地上命中的受創硬直：

```text
弱攻擊      11F
強攻擊      19F
```

並說明防禦的 Hitback 在 Hitstop 結束後立即開始，命中的 Hitback 則在
Hitstop 結束 2F 後開始。這些數字與本專案的 `D2:D3` 逐幀結果完全一致：

| 情境 | 日文資料 | RAM 實測 |
| --- | ---: | ---: |
| 地面弱攻擊防禦 | 9F | `8→0`，9F |
| 地面強攻擊防禦 | 17F | `16→0`，17F |
| 低跳 C/D 防禦 | 9F | `8→0`，9F |
| 正常跳 C/D 防禦 | 17F | `16→0`，17F |
| 地面 CD 防禦 | 21F | `20→0`，21F |
| 地面弱攻擊命中 | 11F | 2F 延遲加 `8→0`，11F |
| 地面強攻擊命中 | 19F | 2F 延遲加 `16→0`，19F |

《技データ解説》另外明確寫道：

```text
GCCD 的防禦硬直與地上ふっとばし攻擊相同，為 21F。
```

因此 `20→0` 應按包含 0 的 21 個狀態解讀為 21F；這不是
inclusive/exclusive 計數造成的落差。

### 6.4 與 SuperCombo Wiki 的來源差異

[SuperCombo Wiki《The King of Fighters '98/System》](https://wiki.supercombo.gg/w/The_King_of_Fighters_%2798/System)
的 `Universal Hitstun and Blockstun` 表格寫：

```text
Jumping Heavy Blockstun = 15F
CD Blockstun            = 23F
```

這組數字同時與上述兩份日文資料及本專案逐幀實測不符。目前應將它視為
來源間的未解差異，而不是本專案 RAM 計時的已知誤差。技術實作優先採用：

1. 本專案可重現的逐幀 RAM trace。
2. 與 trace 完全吻合的《硬直時間》及《技データ解説》。
3. SuperCombo Wiki 保留為衝突的次要參考，等待確認其版本或計數定義。

## 7. Guard Crush 探索性觀察

此項已能自動產生，但尚未固化成正式 regression，先標為探索性資料。

測試方式：

```text
P1 持續防禦
P2 自動接近並反覆使用重攻擊
逐幀監看 Guard Crush、HP、Combo、E3、D2:D3
```

在 Guard Crush 值降到 3 後，下一次接觸觀察到：

```text
Guard Crush：3 → 103
E3 & 0x60：0x60
P1 HP loss：0
Combo delta：0
D2:D3：-1
下一幀 E3 清除
```

這表示：

```text
E3=0x60 並不必然等於一般命中
```

較安全的真實命中條件仍應要求：

```cpp
hit_signature &&
(hp_loss > 0 || combo_delta > 0)
```

Guard Crush 應考慮獨立分類成：

```text
GUARD_CRUSH
```

而不是 `CLEAN_HIT` 或普通 `BLOCK_CONTACT`。

### 7.1 完整對戰 Trace 交叉驗證

`kof98_20260723_135215.csv` 包含 F341～F6423，共 6083 個連續 emulator
frames，沒有缺幀。排除角色／回合切換造成的 HP 與 Guard Crush reset 後，
取得以下結果。

有 Guard signature 且 Guard Crush 確實消耗的 21 次防禦接觸全部符合：

```text
接觸幀
→ Hitstop 11F
→ D2:D3 載入反應倒數
```

| 倒數 | 語意 | 次數 |
| --- | --- | ---: |
| `8→0` | 9F 輕攻擊防禦反應 | 12 |
| `16→0` | 17F 重攻擊防禦反應 | 9 |

21 次接觸全部在 `contact_frame + 11` 載入 `8` 或 `16`，沒有例外。

其中 5 次同時造成削血：

```text
E3 & 0x60 == 0x20
HP loss > 0
Combo 沒有增加
Guard Crush spent > 0
```

總削血為 8 HP。這組資料直接支持 `CHIP_BLOCK` fallback，不應把有 HP loss
的接觸一律分類成 Clean Hit。

排除 reset 後，另外有 53 個造成 HP loss 的接觸幀呈現 Hit signature
(`E3 & 0x60 == 0x60`)。F5427 是同幀互毆：P1、P2 同時掉血且雙方皆有
Hit signature，因此事件系統必須各自產生一次雙向傷害事件，不能互相覆蓋
或把同一側傷害重複加總。

P1 的輸入也提供 Manual／Auto Guard 反例：

```text
F4102/F4198/F4310：站立後防，重攻擊削血防禦
F4406：站立後防，重攻擊無削血防禦
F5526：下後輸入，9F 蹲防
F1058～F1065：持續按後
F1066 接觸幀：輸入已變成 Neutral，但仍發生 Block Contact
```

這證明 provenance 不能只看接觸當幀，必須保留至少前一幀輸入與
`manual_guard_hold_active`。物理 `BLOCK_CONTACT` 與
`MANUAL_BLOCK_SUCCESS` 仍必須分開。P2 在整份 trace 中的 frontend input
都是 `NEUTRAL`，表示 CPU 內部輸入沒有進入目前的 raw joypad logger；
P2 的資料可驗證物理接觸，但不能用來驗證手動防禦。

這份 trace 也證明不能在接觸當幀用 `reaction_counter > 0` 判定倒數開始：

```text
F887 接觸時 D2:D3 = 503，F898 才載入 16
F3154 接觸時 D2:D3 = 11， F3165 才載入 8
F4102 接觸時 D2:D3 = 0，  F4113 才載入 16
```

`D2:D3` 在接觸前可能保留其他反應子階段。Tracker 應由已分類的接觸啟動，
等待 Hitstop 結束後的新倒數載入，不能把任意正值當成 Blockstun。

多段防禦還會刷新既有倒數。例如：

```text
F887 重攻擊接觸 → F898 載入 16
F910 輕攻擊接觸 → F921 重新載入 8
F928 輕攻擊接觸 → F939 重新載入 8
F947 倒數到 0
F948 轉為 -1，Guard signature 清除
```

整條防禦字串只應在最後一次 `0→-1` 時產生一次 Reaction End。

另有一筆尚未能安全分類的 refresh：

```text
F6102：沒有新的 Guard Crush spent 或 HP loss，但 STOP 由 11 變成 2
F6104：D2:D3 由 8 重新載入 16
```

它可能是多段必殺技的後續接觸，但目前缺少可靠的 Hitbox overlap 與
Guard Crush／HP 證據。應保留成 diagnostic，不應直接產生訓練 Reward。

## 8. 建議的接觸分類

分類結果應只有一個主要事件，再由欄位描述削血、Guard Crush 消耗與輸入
來源。不要同時發送 `normal_block` 和 `chip_block` 兩個主要事件。

```cpp
const uint8_t reaction = after.e3 & 0x60;

const bool guard_signature = reaction == 0x20;
const bool hit_signature   = reaction == 0x60;

const bool guard_spent =
    baseline_valid &&
    before.guard_crush >= after.guard_crush &&
    before.guard_crush - after.guard_crush > 0;

const bool chip_damage =
    after.hp_loss > 0 &&
    combo_delta <= 0;

const bool block_fact =
    guard_spent || chip_damage;

const bool block_contact =
    guard_signature &&
    block_fact;

const bool real_hit =
    hit_signature &&
    (hp_loss > 0 || combo_delta > 0);

const bool guard_crush =
    hit_signature &&
    guard_value_reset &&
    hp_loss == 0 &&
    combo_delta <= 0;
```

分類優先序：

```text
Reset／invalid baseline → 不分類
Guard Crush            → GUARD_CRUSH
Block Contact          → BLOCK_CONTACT
Real Hit               → HIT_CONTACT
其他                    → diagnostic／忽略
```

`BLOCK_CONTACT` 應攜帶：

```cpp
enum class GuardProvenance {
    Manual,   // 有正確防禦輸入或已建立 manual guard hold
    Auto,     // 有遊戲 autoguard／其他自動防禦的正面證據
    Unknown,  // 缺少足夠證據，不能由「沒看到 Manual」反推 Auto
};

struct BlockContactInfo {
    bool chip_damage;
    bool guard_spent;
    bool attack_guard_overlap; // 僅為輔助可信度
    int32_t hp_loss;
    int32_t guard_crush_delta;
    GuardProvenance provenance;
};
```

Provenance 與物理 Block 分開判定：

```cpp
if (manual_guard_hold_active ||
    correct_guard_input_now ||
    correct_guard_input_previous_frame) {
    provenance = GuardProvenance::Manual;
} else if (auto_guard_source_confirmed) {
    provenance = GuardProvenance::Auto;
} else {
    provenance = GuardProvenance::Unknown;
}
```

CPU P2 的 frontend input 在完整對戰 trace 中始終為 `NEUTRAL`，因此不能因為
`correct_guard_input == false` 就把 CPU 的 Block 判為 Auto Guard；輸入來源
不可觀察或證據不足時必須標成 `Unknown`。

F1066 接觸幀雖然是 `NEUTRAL`，前幾幀仍可能存在持續後方向，因此 provenance
必須查看 guard hold 與前一幀輸入，不能只讀 contact frame。整條 guard
string 共用同一個 tracker，且 `manual_success_emitted` 保證最多只發一次
`MANUAL_BLOCK_SUCCESS`；後續多段 contact 只更新 block telemetry。

Hitbox overlap 只能當輔助證據，不能當必要條件。既有 trace 中普通技接觸時，
Hitbox overlay 偶爾未捕捉到逐幀 overlap，但 Guard Crush delta 與 E3 仍能
正確分類。

一個接觸只產生一次傷害 delta。F5427 的同幀互毆應產生兩個不同 defender
的 `HIT_CONTACT`，但同一側的 Step Event、HP reward 與 action attribution
不得重複加總。

## 9. 已知限制

1. 測試角色目前只有京。
2. 測試核心是目前專案的 FBNeo build，不代表 MAME、Fightcade 其他版本一定
   有完全相同的 RAM 時序。
3. 正常跳 C/D 與地面 CD 已和兩份日文資料互相驗證；目前只剩
   SuperCombo Wiki 的 15F／23F 是衝突來源。
4. Guard Crush 已能重現，但還缺專用事件與固定 regression。
5. 尚未對投技、Counter Hit、空中受擊全種類、Guard Cancel CD/AB 建立完整
   自動矩陣。
6. `D2:D3=-2` 等負值代表其他反應階段的可能性很高；未驗證前不能命名。
7. P2 CPU 的內部輸入無法由目前 frontend input logger 觀察，因此其 Guard
   provenance 只能標成 `Unknown`。
8. F6102 的無 Guard Crush／HP 證據 counter refresh 尚未完成分類。

## 10. 重跑方式

只跑 C++ 動作、Step Events、RAM timing 與 raw joypad 驗證：

```powershell
C:\Users\User\anaconda3\envs\KofAI\python.exe `
  tools\verify_kof98_actions.py `
  --root F:\Source\qneogeo `
  --skip-scenarios
```

本次完整輸出：

```text
tmp/verify_reaction_extended_20260723.log
```

關鍵通過項：

```text
[PASS] arcade reaction timing
[PASS] guard release actionability
[PASS] raw jump/CD reactions
```

## 11. 希望另一個模型協助比對的問題

1. SuperCombo Wiki 的 Jumping Heavy 15F、CD 23F 為何與兩份日文資料及
   本專案 RAM 實測的 17F、21F 不同？是否使用不同版本或動作分類？
2. 《硬直時間》的 21F 與 `D2:D3 20→0` 已互相驗證；是否還有 MAME
   debugger 或 68000 程式碼可進一步確認 counter 的載入來源？
3. `E0=0x0B` 與 `E0=0x03` 是否可由其他 KOF98 逆向資料分別確認為
   Hop/Normal Jump 狀態？
4. Guard Crush 時 `E3=0x60`、Guard 值重設但沒有 HP/Combo 增量，是否有
   公開 Lua 或 MAME debugger 資料可支持獨立 `GUARD_CRUSH` 分類？
5. 是否存在比 `D2:D3` 更直接的「角色第一個可接受輸入幀」RAM flag？

## 12. 目前技術結論

> E3 提供接觸類型特徵；Guard Crush、HP、Combo delta 與實際輸入提供
> 事實證據；D2:D3 只追蹤已確認情境中的反應子階段。地面防禦在
> 已確認 Block tracker 中發生 `0→-1` 且 Guard E3 回到 `0x00` 的同一個
> engine frame，可接受新的按鍵邊緣。PPO 只能在下一個 policy decision
> boundary 看見並利用該事件。Stale counter、reset、refresh 與未知狀態
> 一律採保守失敗，不產生訓練獎勵。
