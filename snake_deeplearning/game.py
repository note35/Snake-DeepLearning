import pygame
import random
import time


class SnakeGame:
    SPEED = 10
    SIDE = 500
    RED = (255, 100, 100)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)
    CLOCK = pygame.time.Clock() 

    def __init__(self):
        self.head = [250,250]
        self.body = [[250,250],[240,250],[230,250],[220,250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.display = pygame.display.set_mode((self.SIDE, self.SIDE))

    def start_game(self):
        # initial setup
        pygame.init()
        self.display.fill(self.BLACK)
        pygame.display.update()

        # game loop
        self.__play_game(1)

        # game over
        self.display = pygame.display.set_mode((self.SIDE, self.SIDE))
        self.display.fill(self.BLACK)
        pygame.display.update()
        self.__display_score(f'Your Score is: {self.score}')
        pygame.quit()

    def __meet_apple(self):
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score += 1

    def __meet_boundaries(self):
        return 0 if self.SIDE > self.head[0] >= 0 and self.SIDE > self.head[1] >= 0 else 1

    def __meet_self(self):
        return 1 if self.body[0] in self.body[1:] else 0

    def __meet_obstacle(self):
        return 1 if self.__meet_boundaries() == 1 or self.__meet_self() == 1 else 0

    def __generate_snake(self, button_direction):
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

    def __display_snake(self):
        for position in self.body:
            pygame.draw.rect(
                self.display,
                self.WHITE,
                pygame.Rect(position[0], position[1], 10, 10)
            )

    def __display_apple(self):
        # image = pygame.image.load('snake_deeplearning/apple.jpg')
        # self.display.blit(image, (self.apple_position[0], self.apple_position[1]))
        pygame.draw.rect(
            self.display,
            self.YELLOW,
            pygame.Rect(self.apple_position[0], self.apple_position[1], 10, 10)
        )

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
            self.__display_apple()
            self.__display_snake()
            self.__generate_snake(button_direction)

            pygame.display.set_caption(f'Snake Game  SCORE: {self.score}')
            pygame.display.update()

            prev_button_direction = button_direction
            if self.__meet_obstacle() == 1:
                crashed = True
                break

            self.CLOCK.tick(SnakeGame.SPEED)


def main():
    sg = SnakeGame()
    sg.start_game()
