#pragma once

#include "emulatorview.h"
#include "gamememreader.h"

#include <QPoint>

#include <array>
#include <cstdint>

struct GamePlayerState {
    int32_t health = -1;
    int32_t power = -1;
    int32_t power_state = -1;
    int32_t stun = -1;
    QPoint position;
    bool has_position = false;
};

struct GameObservation {
    GamePlayerState p1;
    GamePlayerState p2;
    HitboxOverlay hitbox_overlay;
};

struct GameAction {
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    bool a = false;
    bool b = false;
    bool c = false;
    bool d = false;
};

class IGameAgent {
public:
    virtual ~IGameAgent() = default;

    virtual void reset() = 0;
    virtual GameAction update(const GameObservation &observation) = 0;
protected:
    IGameAgent();
};

class KofAgent final : public IGameAgent {
public:
    void reset() override;
    GameAction update(const GameObservation &observation) override;
    
private:
    int32_t frame_counter_ = 0;
};

GameObservation getGameObservation(const GameMemReader &reader);
std::array<bool, 16> actionToJoypadState(const GameAction &action);
