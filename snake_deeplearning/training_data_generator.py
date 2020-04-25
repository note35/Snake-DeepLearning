from snake_deeplearning.game import SnakeGameForTraining


class TrainingDataGenerator:

    TRAINING_GAMES = 3
    STEPS_PER_GAME = 100

    def __init__(self, game):
        self.game = game
        self.training_data_x = []
        self.training_data_y = []

    def generate_training_data(self):

        for _ in range(TrainingDataGenerator.TRAINING_GAMES):
            self.game.reset_game()
            # prev_apple_distance = self.game.get_apple_distance_from_snake()
            for _ in range(TrainingDataGenerator.STEPS_PER_GAME):
                angle_with_apple, normalized_apple_vec, normalized_snake_vec = self.game.get_angle_with_apple()
                direction, button_direction = self.game.generate_random_direction(angle_with_apple)
                is_front_blocked, is_left_blocked, is_right_blocked = self.game.blocked_directions()

                direction, button_direction = self.__generate_training_data_y(
                    angle_with_apple, button_direction, direction,
                    is_front_blocked, is_left_blocked, is_right_blocked
                )

                if is_front_blocked == 1 and is_left_blocked == 1 and is_right_blocked == 1:
                    break

                self.training_data_x.append([
                    is_left_blocked, is_front_blocked, is_right_blocked,
                    normalized_apple_vec[0], normalized_snake_vec[0],
                    normalized_apple_vec[1], normalized_snake_vec[1]
                ])
                self.game.play_game_by_ai(button_direction)

    def __generate_training_data_y(
        self,
        angle_with_apple, button_direction, direction,
        is_front_blocked, is_left_blocked, is_right_blocked
    ):
        if direction == -1:
            if is_left_blocked == 1:
                if is_front_blocked == 1 and is_right_blocked == 0:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 1)
                    self.training_data_y.append([0, 0, 1])
                elif is_front_blocked == 0 and is_right_blocked == 1:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 0)
                    self.training_data_y.append([0, 1, 0])
                elif is_front_blocked == 0 and is_right_blocked == 0:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 1)
                    self.training_data_y.append([0, 0, 1])
            else:
                self.training_data_y.append([1, 0, 0])
        elif direction == 0:
            if is_front_blocked == 1:
                if is_left_blocked == 1 and is_right_blocked == 0:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 1)
                    self.training_data_y.append([0, 0, 1])
                elif is_left_blocked == 0 and is_right_blocked == 1:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, -1)
                    self.training_data_y.append([1, 0, 0])
                elif is_left_blocked == 0 and is_right_blocked == 0:
                    self.training_data_y.append([0, 0, 1])
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 1)
            else:
                self.training_data_y.append([0, 1, 0])
        else:
            if is_right_blocked == 1:
                if is_left_blocked == 1 and is_front_blocked == 0:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, 0)
                    self.training_data_y.append([0, 1, 0])
                elif is_left_blocked == 0 and is_front_blocked == 1:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, -1)
                    self.training_data_y.append([1, 0, 0])
                elif is_left_blocked == 0 and is_front_blocked == 0:
                    direction, button_direction = self.game.direction_vector(angle_with_apple, -1)
                    self.training_data_y.append([1, 0, 0])
            else:
                self.training_data_y.append([0, 0, 1])

        return direction, button_direction


def main():
    sg = SnakeGameForTraining()
    sg.SPEED = 5000
    tdg = TrainingDataGenerator(sg)
    tdg.generate_training_data()
