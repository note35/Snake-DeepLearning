import math
import numpy as np
import pygame
import random
import time


'''
LEFT -> button_direction = 0
RIGHT -> button_direction = 1
DOWN -> button_direction = 2
UP -> button_direction = 3
'''

class SnakeGame:
    SPEED = 10
    SIDE = 500
    RED = (255, 100, 100)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)
    CLOCK = pygame.time.Clock()
    # OBSTACLES = 0

    def __init__(self):
        self.reset_game()
        self.display = pygame.display.set_mode((self.SIDE, self.SIDE))

    def reset_game(self):
        self.head = [250,250]
        self.body = [[250,250],[240,250],[230,250],[220,250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        # self.obstacles = [[random.randrange(1, 50) * 10, random.randrange(1, 50) * 10] for _ in range(self.OBSTACLES)]
        self.score = 0

    def __start_game(self):
        pygame.init()
        self.display.fill(self.BLACK)
        pygame.display.update()

    def __finish_game(self):
        self.display = pygame.display.set_mode((self.SIDE, self.SIDE))
        self.display.fill(self.BLACK)
        pygame.display.update()
        self.__display_score(f'Your Score is: {self.score}')
        pygame.quit()

    def start_game(self):
        self.__start_game()
        self.__play_game(1)
        self.__finish_game()

    def __meet_apple(self):
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score += 1

    def __meet_boundaries(self, head):
        if self.SIDE > head[0] >= 0 and self.SIDE > head[1] >= 0:
            return 0
        else:
            return 1

    def __meet_self(self, head):
        if head in self.body[1:]:
            return 1
        else:
            return 0

    '''
    def __meet_obstacles(self, head):
        if head in self.obstacles:
            return 1
        else:
            return 0
    '''

    def meet_obstacle(self, current_direction_vec=None):
        if current_direction_vec is None:
            return 1 if self.__meet_boundaries(self.head) or self.__meet_self(self.head) else 0
            # return 1 if self.__meet_boundaries(self.head) or self.__meet_self(self.head) or self.__meet_obstacles(self.head) else 0
        else:
            next_step = self.head + current_direction_vec
            return 1 if self.__meet_boundaries(next_step) or self.__meet_self(next_step.tolist()) else 0
            # return 1 if self.__meet_boundaries(next_step) or self.__meet_self(next_step.tolist()) or self.__meet_obstacles(self.head) else 0

    def _generate_snake(self, button_direction):
        if button_direction == 1:
            self.head[0] += 10
        elif button_direction == 0:
            self.head[0] -= 10
        elif button_direction == 2:
            self.head[1] += 10
        elif button_direction == 3:
            self.head[1] -= 10
        else:
            pass

        if self.head == self.apple_position:
            self.__meet_apple()
            self.body.insert(0, list(self.head))
        else:
            self.body.insert(0, list(self.head))
            self.body.pop()

    def _display_snake(self):
        for position in self.body:
            pygame.draw.rect(
                self.display,
                self.WHITE,
                pygame.Rect(position[0], position[1], 10, 10)
            )

    def _display_apple(self):
        # image = pygame.image.load('snake_deeplearning/apple.jpg')
        # self.display.blit(image, (self.apple_position[0], self.apple_position[1]))
        pygame.draw.rect(
            self.display,
            self.YELLOW,
            pygame.Rect(self.apple_position[0], self.apple_position[1], 10, 10)
        )

    '''
    def _display_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(
                self.display,
                self.RED,
                pygame.Rect(obs[0], obs[1], 10, 10)
            )
    '''

    def __display_score(self, display_text):
        largeText = pygame.font.Font('freesansbold.ttf', 35)
        TextSurf = largeText.render(display_text, True, self.WHITE)
        TextRect = TextSurf.get_rect()
        TextRect.center = (self.SIDE / 2, self.SIDE / 2)
        self.display.blit(TextSurf, TextRect)
        pygame.display.update()
        time.sleep(2)

    def __play_game(self, button_direction):
        crashed = False
        prev_button_direction = 1
        button_direction = 1

        while crashed is not True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and prev_button_direction != 1:
                        button_direction = 0
                    elif event.key == pygame.K_RIGHT and prev_button_direction != 0:
                        button_direction = 1
                    elif event.key == pygame.K_UP and prev_button_direction != 2:
                        button_direction = 3
                    elif event.key == pygame.K_DOWN and prev_button_direction != 3:
                        button_direction = 2
                    else:
                        button_direction = button_direction
        
            self.display.fill(self.BLACK)
            self._display_apple()
            # self._display_obstacles()
            self._display_snake()
            self._generate_snake(button_direction)

            pygame.display.set_caption(f'Snake Game  SCORE: {self.score}')
            pygame.display.update()

            prev_button_direction = button_direction
            if self.meet_obstacle() == 1:
                crashed = True
                break

            self.CLOCK.tick(SnakeGame.SPEED)


class SnakeGameForTraining(SnakeGame):
    """ Function used by training_data_generator.py """

    def blocked_directions(self):
        current_direction_vec = np.array(self.body[0]) - np.array(self.body[1])
        left_direction_vec = np.array([current_direction_vec[1], -current_direction_vec[0]])
        right_direction_vec = np.array([-current_direction_vec[1], current_direction_vec[0]])

        is_front_blocked = self.meet_obstacle(current_direction_vec)
        is_left_blocked = self.meet_obstacle(left_direction_vec)
        is_right_blocked = self.meet_obstacle(right_direction_vec)

        return current_direction_vec, is_front_blocked, is_left_blocked, is_right_blocked

    def get_angle_with_apple(self):
        apple_direction_vec = np.array(self.apple_position) - np.array(self.body[0])
        snake_direction_vec = np.array(self.body[0]) - np.array(self.body[1])

        norm_apple_direction_vec = np.linalg.norm(apple_direction_vec)
        norm_snake_direction_vector = np.linalg.norm(snake_direction_vec)

        if norm_apple_direction_vec == 0:
            norm_apple_direction_vec = 10
        if norm_snake_direction_vector == 0:
            norm_snake_direction_vector = 10

        normalized_apple_vec = apple_direction_vec / norm_apple_direction_vec
        normalized_snake_vec = snake_direction_vec / norm_snake_direction_vector
        angle = math.atan2(
            normalized_apple_vec[1] * normalized_snake_vec[0] - normalized_apple_vec[0] * normalized_snake_vec[1],
            normalized_apple_vec[1] * normalized_snake_vec[1] + normalized_apple_vec[0] * normalized_snake_vec[0]
        ) / math.pi
        return angle, normalized_apple_vec, normalized_snake_vec

    def generate_random_direction(self, angle_with_apple):
        direction = 0
        if angle_with_apple > 0:
            direction = 1
        elif angle_with_apple < 0:
            direction = -1
        else:
            direction = 0
        return self.direction_vector(angle_with_apple, direction)

    def generate_button_direction(self, new_direction):
        if new_direction == [10, 0]:
            return 1
        elif new_direction == [-10, 0]:
            return 0
        elif new_direction == [0, 10]:
            return 2
        else:
            return 3

    def direction_vector(self, angle_with_apple, direction):
        current_direction_vec = np.array(self.body[0]) - np.array(self.body[1])
        left_direction_vec = np.array([current_direction_vec[1], -current_direction_vec[0]])
        right_direction_vec = np.array([-current_direction_vec[1], current_direction_vec[0]])

        new_direction = current_direction_vec
        if direction == -1:
            new_direction = left_direction_vec
        if direction == 1:
            new_direction = right_direction_vec

        button_direction = self.generate_button_direction(new_direction.tolist())
        return direction, button_direction

    def play_game_by_ai(self, button_direction):
        crashed = False
        while crashed is not True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    break
            
            self.display.fill(self.BLACK)
            self._display_apple()
            # self._display_obstacles()
            self._display_snake()
            self._generate_snake(button_direction)

            pygame.display.set_caption(f'Snake Game  SCORE: {self.score}')
            pygame.display.update()
            self.CLOCK.tick(SnakeGame.SPEED)
            break


def main():
    sg = SnakeGame()
    sg.start_game()
