#include "gameai.h"

#include <cstdlib>
#include <utility>

#include "libretro.h"

IGameAgent::IGameAgent() = default;

void KofAgent::reset() {
    frame_counter_ = 0;
}

GameAction KofAgent::update(const GameObservation &observation) {
    ++frame_counter_;

    GameAction action;
    if (!observation.p1.has_position || !observation.p2.has_position)
        return action;

    const int32_t distance = observation.p2.position.x() - observation.p1.position.x();
    const int32_t abs_distance = std::abs(distance);

    if (abs_distance > 52) {
        action.left = distance < 0;
        action.right = distance > 0;
        return action;
    }

    if ((frame_counter_ / 8) % 2 == 0)
        action.c = true;

    return action;
}

GameObservation getGameObservation(const GameMemReader &reader) {
    GameObservation observation;
    observation.hitbox_overlay  = reader.getHitboxOverlay();

    observation.p1.health       = reader.readP1Health();
    observation.p1.power        = reader.readP1Power();
    observation.p1.power_state  = reader.readP1PowerState();
    observation.p1.stun         = reader.readP1Stun();
    observation.p1.has_position = reader.readP1Position(observation.p1.position);

    observation.p2.health       = reader.readP2Health();
    observation.p2.power        = reader.readP2Power();
    observation.p2.power_state  = reader.readP2PowerState();
    observation.p2.stun         = reader.readP2Stun();
    observation.p2.has_position = reader.readP2Position(observation.p2.position);

    return observation;
}

std::array<bool, 16> actionToJoypadState(const GameAction &action) {
    std::array<bool, 16> state {};
    state[RETRO_DEVICE_ID_JOYPAD_UP] = action.up;
    state[RETRO_DEVICE_ID_JOYPAD_DOWN] = action.down;
    state[RETRO_DEVICE_ID_JOYPAD_LEFT] = action.left;
    state[RETRO_DEVICE_ID_JOYPAD_RIGHT] = action.right;
    state[RETRO_DEVICE_ID_JOYPAD_B] = action.a;
    state[RETRO_DEVICE_ID_JOYPAD_A] = action.b;
    state[RETRO_DEVICE_ID_JOYPAD_Y] = action.c;
    state[RETRO_DEVICE_ID_JOYPAD_X] = action.d;
    return state;
}
