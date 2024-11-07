#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
from typing import Type

import pygame
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

import matplotlib.pyplot as plt
import numpy as np


class FuzzyPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)
        # for Mamdami:
        x_dist = fuzzcontrol.Antecedent(np.arange(-800, 801, 1), 'x_distance')
        y_dist = fuzzcontrol.Antecedent(np.arange(0, 401, 1), 'y_distance')
        velocity = fuzzcontrol.Consequent(np.arange(-10, 11, 1), 'velocity')


        x_dist['far_right'] = fuzz.trimf(x_dist.universe, [-800, -800, -250])
        x_dist['right'] = fuzz.trimf(x_dist.universe, [-700, -50, 0])
        x_dist['center'] = fuzz.trimf(x_dist.universe, [-20, 0, 20])
        x_dist['left'] = fuzz.trimf(x_dist.universe, [0, 50, 700])
        x_dist['far_left'] = fuzz.trimf(x_dist.universe, [250, 800, 800])

        y_dist['very_close'] = fuzz.trimf(y_dist.universe, [0, 0, 200])
        y_dist['close'] = fuzz.trimf(y_dist.universe, [80, 200, 300])
        y_dist['medium'] = fuzz.trimf(y_dist.universe, [200, 300, 350])
        y_dist['far'] = fuzz.trimf(y_dist.universe, [300, 400, 400])

        velocity['fast_left'] = fuzz.trimf(velocity.universe, [-10, -10, -6])
        velocity['medium_left'] = fuzz.trimf(velocity.universe, [-8, -6, -3])
        velocity['slow_left'] = fuzz.trimf(velocity.universe, [-5, -3, 0])
        velocity['stop'] = fuzz.trimf(velocity.universe, [-2, 0, 2])
        velocity['slow_right'] = fuzz.trimf(velocity.universe, [0, 3, 5])
        velocity['medium_right'] = fuzz.trimf(velocity.universe, [3, 6, 8])
        velocity['fast_right'] = fuzz.trimf(velocity.universe, [6, 10, 10])


        rule1 = fuzzcontrol.Rule(x_dist['far_left'] & y_dist['very_close'], velocity['fast_left'])
        rule2 = fuzzcontrol.Rule(x_dist['far_left'] & y_dist['close'], velocity['fast_left'])
        rule3 = fuzzcontrol.Rule(x_dist['far_left'] & y_dist['medium'], velocity['medium_left'])
        rule4 = fuzzcontrol.Rule(x_dist['far_left'] & y_dist['far'], velocity['slow_left'])

        rule5 = fuzzcontrol.Rule(x_dist['left'] & y_dist['very_close'], velocity['medium_left'])
        rule6 = fuzzcontrol.Rule(x_dist['left'] & y_dist['close'], velocity['medium_left'])
        rule7 = fuzzcontrol.Rule(x_dist['left'] & y_dist['medium'], velocity['slow_left'])
        rule8 = fuzzcontrol.Rule(x_dist['left'] & y_dist['far'], velocity['slow_left'])

        rule9  = fuzzcontrol.Rule(x_dist['center'] & y_dist['very_close'], velocity['stop'])
        rule10 = fuzzcontrol.Rule(x_dist['center'] & y_dist['close'], velocity['stop'])
        rule11 = fuzzcontrol.Rule(x_dist['center'] & y_dist['medium'], velocity['stop'])
        rule12 = fuzzcontrol.Rule(x_dist['center'] & y_dist['far'], velocity['stop'])

        rule13 = fuzzcontrol.Rule(x_dist['right'] & y_dist['very_close'], velocity['medium_right'])
        rule14 = fuzzcontrol.Rule(x_dist['right'] & y_dist['close'], velocity['medium_right'])
        rule15 = fuzzcontrol.Rule(x_dist['right'] & y_dist['medium'], velocity['slow_right'])
        rule16 = fuzzcontrol.Rule(x_dist['right'] & y_dist['far'], velocity['slow_right'])
        
        rule17 = fuzzcontrol.Rule(x_dist['far_right'] & y_dist['very_close'], velocity['fast_right'])
        rule18 = fuzzcontrol.Rule(x_dist['far_right'] & y_dist['close'], velocity['fast_right'])
        rule19 = fuzzcontrol.Rule(x_dist['far_right'] & y_dist['medium'], velocity['medium_right'])
        rule20 = fuzzcontrol.Rule(x_dist['far_right'] & y_dist['far'], velocity['slow_right'])
        



        self.racket_controller = fuzzcontrol.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20
        ])
        self.fuzzy_ctrl = fuzzcontrol.ControlSystemSimulation(self.racket_controller)

        # visualize Mamdami
        # x_dist.view()
        # y_dist.view()
        # velocity.view()

        # for TSK:
        # self.x_universe = np.arange...
        # self.x_mf = {
        #     "far_left": fuzz.trapmf(
        #         self.x_universe,
        #         [
        #             ...
        #         ],
        #     ),
        #     ...
        # }
        # ...
        # self.velocity_fx = {
        #     "f_slow_left": lambda x_diff, y_diff: -1 * (abs(x_diff) + y_diff),
        #     ...
        # }

        # visualize TSK
        # plt.figure()
        # for name, mf in self.x_mf.items():
        #     plt.plot(self.x_universe, mf, label=name)
        # plt.legend()
        # plt.show()
        # ...

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):
        # for Mamdami:
        self.fuzzy_ctrl.input['x_distance'] = x_diff
        self.fuzzy_ctrl.input['y_distance'] = y_diff
        self.fuzzy_ctrl.compute()
        
        velocity = self.fuzzy_ctrl.output['velocity']
        # print(f"X distance: {x_diff}")
        # print(f"Y distance: {y_diff}")
        # print(f"Velocity: {velocity}")
        
        return velocity


        # for TSK:
        # x_vals = {
        #     name: fuzz.interp_membership(self.x_universe, mf, x_diff)
        #     for name, mf in self.x_mf.items()
        # }
        # ...
        # rule activations with Zadeh norms
        # activations = {
        #     "f_slow_left": max(
        #         [
        #             min(x_vals...),
        #             min(x_vals...),
        #         ]
        #     ),
        #     ...
        # }

        # velocity = sum(
        #     activations[val] * self.velocity_fx[val](x_diff, y_diff)
        #     for val in activations
        # ) / sum(activations[val] for val in activations)

        return 0


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    game.run()
