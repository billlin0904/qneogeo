#include "koframtracelogger.h"

#include "gamememreadercore.h"

#include <QDir>
#include <QFileInfo>

#include <algorithm>
#include <array>
#include <cstdlib>

namespace {
constexpr uint32_t INPUT_UP = 1u << 0;
constexpr uint32_t INPUT_DOWN = 1u << 1;
constexpr uint32_t INPUT_LEFT = 1u << 2;
constexpr uint32_t INPUT_RIGHT = 1u << 3;
constexpr uint32_t INPUT_A = 1u << 4;
constexpr uint32_t INPUT_B = 1u << 5;
constexpr uint32_t INPUT_C = 1u << 6;
constexpr uint32_t INPUT_D = 1u << 7;
constexpr uint64_t FLUSH_INTERVAL_FRAMES = 60;

struct PlayerHitboxFacts {
    int32_t attack_count = 0;
    int32_t guard_count = 0;
    int32_t hurt_count = 0;
};

struct HitboxFacts {
    std::array<PlayerHitboxFacts, 2> players;
    bool p1_attack_guard_overlap = false;
    bool p2_attack_guard_overlap = false;
    bool p1_attack_hurt_overlap = false;
    bool p2_attack_hurt_overlap = false;
};

uint32_t inputMask(const EmulatorView::JoypadInput &input) {
    uint32_t result = 0;
    if (input.up)
        result |= INPUT_UP;
    if (input.down)
        result |= INPUT_DOWN;
    if (input.left)
        result |= INPUT_LEFT;
    if (input.right)
        result |= INPUT_RIGHT;
    if (input.a)
        result |= INPUT_A;
    if (input.b)
        result |= INPUT_B;
    if (input.c)
        result |= INPUT_C;
    if (input.d)
        result |= INPUT_D;
    return result;
}

QByteArray inputText(const EmulatorView::JoypadInput &input) {
    QByteArray result;
    auto append = [&result](const char *name) {
        if (!result.isEmpty())
            result += '+';
        result += name;
    };

    if (input.up)
        append("UP");
    if (input.down)
        append("DOWN");
    if (input.left)
        append("LEFT");
    if (input.right)
        append("RIGHT");
    if (input.a)
        append("A");
    if (input.b)
        append("B");
    if (input.c)
        append("C");
    if (input.d)
        append("D");
    return result.isEmpty() ? QByteArrayLiteral("NEUTRAL") : result;
}

bool isAttackBox(int32_t type) {
    return type == game_memory::HitboxAttack ||
           type == game_memory::HitboxProjectileAttack;
}

bool isHurtBox(int32_t type) {
    return type == game_memory::HitboxVulnerability ||
           type == game_memory::HitboxProjectileVulnerability;
}

bool rectanglesOverlap(const game_memory::HitboxRect &lhs,
                       const game_memory::HitboxRect &rhs) {
    return lhs.left < rhs.left + rhs.width &&
           rhs.left < lhs.left + lhs.width &&
           lhs.top < rhs.top + rhs.height &&
           rhs.top < lhs.top + lhs.height;
}

bool boxesOverlap(const game_memory::HitboxOverlay &overlay,
                  int32_t attacker_owner,
                  int32_t target_owner,
                  int32_t target_type) {
    for (const game_memory::HitboxRect &attack : overlay.boxes) {
        if (attack.owner != attacker_owner || !isAttackBox(attack.type))
            continue;

        for (const game_memory::HitboxRect &target : overlay.boxes) {
            if (target.owner != target_owner)
                continue;

            const bool target_matches = target_type == game_memory::HitboxGuard
                ? target.type == game_memory::HitboxGuard
                : isHurtBox(target.type);
            if (target_matches && rectanglesOverlap(attack, target))
                return true;
        }
    }
    return false;
}

HitboxFacts collectHitboxFacts(const game_memory::HitboxOverlay &overlay) {
    HitboxFacts result;
    for (const game_memory::HitboxRect &box : overlay.boxes) {
        if (box.owner < 1 || box.owner > 2)
            continue;

        PlayerHitboxFacts &player = result.players[static_cast<size_t>(box.owner - 1)];
        if (isAttackBox(box.type))
            ++player.attack_count;
        else if (box.type == game_memory::HitboxGuard)
            ++player.guard_count;
        else if (isHurtBox(box.type))
            ++player.hurt_count;
    }

    result.p1_attack_guard_overlap = boxesOverlap(
        overlay, 1, 2, game_memory::HitboxGuard);
    result.p2_attack_guard_overlap = boxesOverlap(
        overlay, 2, 1, game_memory::HitboxGuard);
    result.p1_attack_hurt_overlap = boxesOverlap(
        overlay, 1, 2, game_memory::HitboxVulnerability);
    result.p2_attack_hurt_overlap = boxesOverlap(
        overlay, 2, 1, game_memory::HitboxVulnerability);
    return result;
}

bool holdingBack(const EmulatorView::JoypadInput &input, bool facing_left) {
    return facing_left ? input.right : input.left;
}

bool isD2ZeroPhase(int32_t value) {
    return value == 0;
}

void appendField(QByteArray &row, const QByteArray &value) {
    if (!row.isEmpty())
        row += ',';
    row += value;
}

void appendField(QByteArray &row, int64_t value) {
    appendField(row, QByteArray::number(value));
}

void appendField(QByteArray &row, int32_t value) {
    appendField(row, static_cast<int64_t>(value));
}

void appendField(QByteArray &row, bool value) {
    appendField(row, value ? QByteArrayLiteral("1") : QByteArrayLiteral("0"));
}

void appendPlayerDiagnostics(
    QByteArray &row,
    const game_memory::PlayerReactionDebugState &state,
    int32_t previous_guard_crush,
    int32_t previous_d2,
    int32_t previous_e3,
    bool has_previous_frame) {
    const bool d2_zero_started = has_previous_frame &&
        !isD2ZeroPhase(previous_d2) && isD2ZeroPhase(state.reaction_d2);
    const bool d2_zero_finished = has_previous_frame &&
        isD2ZeroPhase(previous_d2) && !isD2ZeroPhase(state.reaction_d2);
    const bool e3_bit20 = state.reaction_e3 >= 0 && (state.reaction_e3 & 0x20) != 0;
    const bool previous_e3_bit20 = previous_e3 >= 0 && (previous_e3 & 0x20) != 0;

    appendField(row, state.hit_guard_stop);
    appendField(row, state.reaction_d2);
    appendField(row, state.reaction_d3);
    appendField(row, state.reaction_d2d3_unsigned);
    appendField(row, state.reaction_d2d3_signed);
    appendField(row, d2_zero_started);
    appendField(row, d2_zero_finished);
    appendField(row, state.reaction_e0);
    appendField(row, state.reaction_e1);
    appendField(row, state.reaction_e2);
    appendField(row, state.reaction_e3);
    appendField(row, e3_bit20);
    appendField(row, has_previous_frame && !previous_e3_bit20 && e3_bit20);
    appendField(row, has_previous_frame && previous_e3_bit20 && !e3_bit20);
    appendField(row, state.d4_high);
    appendField(row, state.d5_low);
    appendField(row, state.d4_signed);
    appendField(row, state.recovery_control_e7);
    appendField(row, state.guard_crush);
    appendField(row,
                has_previous_frame && previous_guard_crush >= 0 && state.guard_crush >= 0
                    ? previous_guard_crush - state.guard_crush
                    : 0);
}

constexpr const char *CSV_HEADER =
    "frame,round_time,"
    "p1_input_mask,p1_input_text,p2_input_mask,p2_input_text,"
    "p1_x,p1_y,p2_x,p2_y,distance_x,"
    "p1_facing_left,p2_facing_left,p1_holding_back,p2_holding_back,"
    "p1_holding_down_back,p2_holding_down_back,p1_ready,p2_ready,"
    "p1_hp,p2_hp,p1_hp_loss,p2_hp_loss,p1_combo,p2_combo,"
    "p1_power,p2_power,p1_power_stocks,p2_power_stocks,p1_stun,p2_stun,"
    "p1_hitbox_mask,p2_hitbox_mask,p1_attack_boxes,p2_attack_boxes,"
    "p1_guard_boxes,p2_guard_boxes,p1_hurt_boxes,p2_hurt_boxes,"
    "p1_attack_guard_overlap,p2_attack_guard_overlap,"
    "p1_attack_hurt_overlap,p2_attack_hurt_overlap,"
    "p1_manual_block_candidate,p2_manual_block_candidate,"
    "p1_stop,p1_d2,p1_d3,p1_d2d3_unsigned,p1_d2d3_signed,"
    "p1_d2_zero_started,p1_d2_zero_finished,"
    "p1_e0,p1_e1,p1_e2,p1_e3,p1_e3_bit20,p1_e3_bit20_started,p1_e3_bit20_finished,"
    "p1_d4,p1_d5,p1_d4_signed,p1_e7,p1_guard_crush,p1_guard_crush_spent,"
    "p2_stop,p2_d2,p2_d3,p2_d2d3_unsigned,p2_d2d3_signed,"
    "p2_d2_zero_started,p2_d2_zero_finished,"
    "p2_e0,p2_e1,p2_e2,p2_e3,p2_e3_bit20,p2_e3_bit20_started,p2_e3_bit20_finished,"
    "p2_d4,p2_d5,p2_d4_signed,p2_e7,p2_guard_crush,p2_guard_crush_spent\n";
} // namespace

KofRamTraceLogger::~KofRamTraceLogger() {
    stop();
}

bool KofRamTraceLogger::start(const QString &file_path, QString &error) {
    stop();

    const QFileInfo file_info(file_path);
    if (!QDir().mkpath(file_info.absolutePath())) {
        error = QStringLiteral("無法建立記錄資料夾：%1")
                    .arg(QDir::toNativeSeparators(file_info.absolutePath()));
        return false;
    }

    file_.setFileName(file_info.absoluteFilePath());
    if (!file_.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        error = QStringLiteral("無法開啟 RAM 記錄檔：%1")
                    .arg(file_.errorString());
        return false;
    }

    const QByteArray header(CSV_HEADER);
    if (file_.write(header) != header.size()) {
        error = QStringLiteral("無法寫入 RAM 記錄檔：%1")
                    .arg(file_.errorString());
        stop();
        return false;
    }

    previous_p1_ = {};
    previous_p2_ = {};
    has_previous_frame_ = false;
    rows_since_flush_ = 0;
    error.clear();
    return true;
}

void KofRamTraceLogger::stop() {
    if (file_.isOpen()) {
        file_.flush();
        file_.close();
    }

    has_previous_frame_ = false;
    rows_since_flush_ = 0;
}

bool KofRamTraceLogger::isActive() const {
    return file_.isOpen();
}

QString KofRamTraceLogger::filePath() const {
    return file_.fileName();
}

bool KofRamTraceLogger::writeFrame(
    uint64_t frame_number,
    const QByteArray &ram,
    QSize source_size,
    const EmulatorView::JoypadInput &p1_input,
    const EmulatorView::JoypadInput &p2_input,
    QString &error) {
    if (!file_.isOpen()) {
        error = QStringLiteral("RAM 記錄檔尚未開啟。");
        return false;
    }

    const game_memory::GameMemReaderCore reader(
        reinterpret_cast<const uint8_t *>(ram.constData()),
        static_cast<size_t>(ram.size()),
        source_size.width());
    const game_memory::PlayerReactionDebugState p1 = reader.readP1ReactionDebugState();
    const game_memory::PlayerReactionDebugState p2 = reader.readP2ReactionDebugState();
    const int32_t p1_health = reader.readP1Health();
    const int32_t p2_health = reader.readP2Health();

    game_memory::Point p1_position { -1, -1 };
    game_memory::Point p2_position { -1, -1 };
    reader.readP1Position(p1_position);
    reader.readP2Position(p2_position);

    bool p1_facing_left = false;
    bool p2_facing_left = false;
    const bool p1_facing_valid = reader.readP1FacingLeft(p1_facing_left);
    const bool p2_facing_valid = reader.readP2FacingLeft(p2_facing_left);

    const HitboxFacts hitboxes = collectHitboxFacts(reader.getHitboxOverlay());
    const bool p1_holding_back = p1_facing_valid && holdingBack(p1_input, p1_facing_left);
    const bool p2_holding_back = p2_facing_valid && holdingBack(p2_input, p2_facing_left);

    QByteArray row;
    row.reserve(1024);
    appendField(row, static_cast<int64_t>(frame_number));
    appendField(row, reader.readRoundTime());
    appendField(row, static_cast<int64_t>(inputMask(p1_input)));
    appendField(row, inputText(p1_input));
    appendField(row, static_cast<int64_t>(inputMask(p2_input)));
    appendField(row, inputText(p2_input));
    appendField(row, p1_position.x);
    appendField(row, p1_position.y);
    appendField(row, p2_position.x);
    appendField(row, p2_position.y);
    appendField(row, std::abs(p1_position.x - p2_position.x));
    appendField(row, p1_facing_valid && p1_facing_left);
    appendField(row, p2_facing_valid && p2_facing_left);
    appendField(row, p1_holding_back);
    appendField(row, p2_holding_back);
    appendField(row, p1_holding_back && p1_input.down);
    appendField(row, p2_holding_back && p2_input.down);
    appendField(row, reader.p1ReadyForAction());
    appendField(row, reader.p2ReadyForAction());
    appendField(row, p1_health);
    appendField(row, p2_health);
    appendField(row,
                has_previous_frame_ && previous_p1_.health >= 0 && p1_health >= 0
                    ? std::max(0, previous_p1_.health - p1_health)
                    : 0);
    appendField(row,
                has_previous_frame_ && previous_p2_.health >= 0 && p2_health >= 0
                    ? std::max(0, previous_p2_.health - p2_health)
                    : 0);
    appendField(row, reader.readP1ComboCount());
    appendField(row, reader.readP2ComboCount());
    appendField(row, reader.readP1AdvancedPowerValue());
    appendField(row, reader.readP2AdvancedPowerValue());
    appendField(row, reader.readP1AdvancedPowerStocks());
    appendField(row, reader.readP2AdvancedPowerStocks());
    appendField(row, reader.readP1Stun());
    appendField(row, reader.readP2Stun());
    appendField(row, reader.readP1HitboxActiveMask());
    appendField(row, reader.readP2HitboxActiveMask());
    appendField(row, hitboxes.players[0].attack_count);
    appendField(row, hitboxes.players[1].attack_count);
    appendField(row, hitboxes.players[0].guard_count);
    appendField(row, hitboxes.players[1].guard_count);
    appendField(row, hitboxes.players[0].hurt_count);
    appendField(row, hitboxes.players[1].hurt_count);
    appendField(row, hitboxes.p1_attack_guard_overlap);
    appendField(row, hitboxes.p2_attack_guard_overlap);
    appendField(row, hitboxes.p1_attack_hurt_overlap);
    appendField(row, hitboxes.p2_attack_hurt_overlap);
    appendField(row, hitboxes.p2_attack_guard_overlap && p1_holding_back);
    appendField(row, hitboxes.p1_attack_guard_overlap && p2_holding_back);
    appendPlayerDiagnostics(row,
                            p1,
                            previous_p1_.guard_crush,
                            previous_p1_.reaction_d2,
                            previous_p1_.reaction_e3,
                            has_previous_frame_);
    appendPlayerDiagnostics(row,
                            p2,
                            previous_p2_.guard_crush,
                            previous_p2_.reaction_d2,
                            previous_p2_.reaction_e3,
                            has_previous_frame_);
    row += '\n';

    if (file_.write(row) != row.size()) {
        error = QStringLiteral("寫入 RAM 記錄檔失敗：%1")
                    .arg(file_.errorString());
        return false;
    }

    previous_p1_ = { p1_health, p1.guard_crush, p1.reaction_d2, p1.reaction_e3 };
    previous_p2_ = { p2_health, p2.guard_crush, p2.reaction_d2, p2.reaction_e3 };
    has_previous_frame_ = true;

    ++rows_since_flush_;
    if (rows_since_flush_ >= FLUSH_INTERVAL_FRAMES) {
        if (!file_.flush()) {
            error = QStringLiteral("同步 RAM 記錄檔失敗：%1")
                        .arg(file_.errorString());
            return false;
        }
        rows_since_flush_ = 0;
    }

    error.clear();
    return true;
}
