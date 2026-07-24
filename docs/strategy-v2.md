# KOF98 StrategyV2

StrategyV2 保留原本的 29 個高階 Action 與 C++ 逐幀輸入 runtime，避免把已完成的
Combo 技能丟掉。這次只擴充可觀察狀態、清理完整對戰 reward，並提供無損模型遷移。

## Observation ABI

- `v1`: 26 維，所有既有 checkpoint 的固定 ABI。
- `v2`: 140 維，前 26 欄完全不動，後面加入實測或可靠推導的資料。
- 新欄位包含 raw player status、ready/airborne、雙方速度、角落距離、Action
  lifecycle、Action one-hot 與 combat phase。
- 對手的 scripted Action id／elapsed／remaining 不放進 policy。這些欄位在
  140 維 ABI 中保留位置但固定為中性值；原 `p2_scripted` 欄改為部署時也存在的
  `profile_is_fight`，因此舊 V2 checkpoint 可直接續訓。
- 不把 raw status 猜成 hitstun、blockstun 或 invulnerability；那些語意需先用
  Memory View 與 deterministic case 驗證。

## Model migration

`--migrate-from` 建立新的 140 維 policy，完整複製同形狀 tensor。Policy/value 的
第一層會先清零，再把舊 26 欄權重複製過去。因此遷移瞬間：

```text
V2 observation[26:] = 0
=> V2 logits == V1 logits
=> V2 value  == V1 value
```

Optimizer 重新建立。`--teacher-model` 在前段 rollout 加入會線性退火的 policy
distillation loss，防止 PPO 在新 reward／新 observation 下立刻忘記舊技能。

## Full Fight reward

StrategyV2 使用：

```text
10 * (damage_to_P2 - damage_to_P1) / 103
+10 win / -10 loss
```

完整 Fight 不再持續支付距離、hitbox overlap、防禦、招式名稱、快速勝利或 combo
milestone。這些訊號若要使用，只應出現在有明確成功條件的 tactical curriculum。
Combo Profile 的 phase reward 不變，仍負責教招式如何執行。

## Paper-derived decisions

- *Policy Invariance Under Reward Transformations*：完整 Fight 不使用可反覆累積的
  距離 shaping；未來課程若需要距離引導，應使用 potential difference。
- *Kickstarting Deep Reinforcement Learning*：模型遷移後使用退火 teacher loss，
  保留已訓練技能但允許 student 最終超越 teacher。
- *Using Reward Machines for High-Level Task Specification*：combat phase 是明確的
  有限狀態 observation，不直接支付永久 reward。
- *Deep Reinforcement Learning with Action Delay* 與 semi-MDP 文獻：Action lifecycle
  暴露 elapsed/remaining frame；仍維持固定 repeat4，避免可變 step 造成 discount
  語意不一致。
- FightingICE PPO 與商業格鬥遊戲研究：加入速度、可行動狀態、目前 Action 與剩餘
  frame，讓 policy 能區分接近、確認、連段、倒地等局面。

PLR、reverse curriculum 與 frozen-opponent self-play 尚未塞入這一版。它們需要
state bank 與 opponent checkpoint pool；在 StrategyV2 基線通過 holdout bot 與真人
測試前加入，只會讓問題來源更難分辨。

## Commands

VS Code：

```text
Train StrategyV2 (Migrate + Kickstart)
Train StrategyV2 (Tactics Fine-tune)
Watch StrategyV2 Fight
Verify KOF98 Actions
```

正式訓練前應先執行 `Verify KOF98 Actions`。TensorBoard 另外監看：

```text
kickstart/weight
kickstart/loss
kof/reward_hp_total
kof/reward_outcome_total
kof_fight_physical/*/win_rate
kof_fight_physical/*/combo_4plus_episode_rate
kof_fight_targeted/defense/success_rate
kof_fight_targeted/anti_air/success_rate
kof_fight_targeted/approach/success_rate
kof_fight_targeted/hit_confirm/success_rate
kof_fight_physical/action_1_free_decision_rate
kof_fight_physical/action_2_free_decision_rate
kof_fight_physical/action_5_free_decision_rate
kof_fight_physical/action_16_free_decision_rate
kof_fight_physical/crouch_b_close_c_free_decision_rate
kof_fight_physical/guard_given_pressure_rate
kof_fight_physical/oniyaki_hit_given_airborne_rate
kof_fight_physical/safe_entry_given_far_rate
kof_fight_physical/followup_hit_given_confirm_rate
```

## Tactics fine-tune

`Train StrategyV2 (Tactics Fine-tune)` 從已完成的
`kof98_strategy_v2_ppo_final.zip` 接續 150 萬 steps，不使用舊 teacher model。
為避免每個 FBNeo process 的完整 ROM/RAM 配置耗盡主記憶體，10 個常駐環境分成：

```text
2 Combo：每次 reset 輪替全部 11 套連段
2 Targeted Fight：每次 reset 輪替防禦/確反、對空、接近、hit confirm
6 Physical Fight：每次 reset 輪替四種 P2 style
```

三類樣本比例仍是 20%／20%／60%，因此 150 萬 steps 的課程權重不變；減少的是
同時存在的模擬器實例，而不是任何課程或對手風格。

Targeted curriculum 只在對應機會出現時限制合法 Action，主 Fight reward 仍維持
對稱傷害與勝負，不為特定招式永久加分。TensorBoard 的
`kof_fight_targeted/*` 是上課成績；泛化成績仍以 `kof_fight_physical/*` 為準。
防禦課在壓力中只開放防禦，成功後開 36-frame 確反窗，候選為近 C、蹲 B、
鬼燒；`defense/guard_success_rate` 與 `defense/counter_success_rate` 分開記錄。
關鍵 Action rate 也按 mode 拆開，不能拿 targeted teacher 強迫產生的動作當成
physical policy 已經學會的證據。`*_free_decision_rate` 只以 DLL 真正空閒、policy
可以自由選招的 step 為分母，不會被輸入腳本播放期間的 forced Idle 稀釋。

四個條件式成功率使用實際事件而非「有按到 Action」：防禦要求 P2 實際攻擊且
P1 無傷，對空要求鬼燒實際命中空中的 P2，接近要求由遠距離無傷進入 90 px，
hit confirm 要求近 C／蹲 B 起手實際命中後，已 queue 的後續招也實際命中。
這些計數跨 rollout 累積，避免機會與成功剛好落在 rollout 邊界時扭曲比例。

真人驗收要勾選 `Tools > P2 AI Pure Policy`。此模式只讓單一 Fight policy 決策，
完全停用 Combo model 切換與遠距離蹲 A／蹲 B override；未勾選的混合模式只供
遊玩與展示，不可拿來宣稱 policy 本身的能力。

Tactics fine-tune 的最低驗收線以 Physical 指標為準：勝率至少 85%、4+ combo
至少 8%、Action 11 的自由決策占比低於 60%、Action 11 加 Action 8 低於 80%，
且四種 P2 style 的最低勝率至少 70%。Targeted 成功但 Physical 條件成功率沒有
上升，代表仍依賴老師 mask，不能算戰術已轉移。
