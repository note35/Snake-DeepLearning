import json

from keras.models import model_from_json, Sequential
from keras.layers import Dense
import numpy as np

from snake_deeplearning.game import SnakeGameForTraining


class TrainingDataGenerator:

    def __init__(self, game, training_games, steps_per_game):
        self.game = game
        self.training_data_x = []  # input
        self.training_data_y = []  # output
        self.training_games = training_games
        self.steps_per_game = steps_per_game
        self.model_name = f'model_{training_games}_{steps_per_game}'

    def generate_training_data(self):
        for _ in range(1, self.training_games+1):
            print(f'GAME {_}')
            self.game.reset_game()
            for _ in range(self.steps_per_game):
                angle_with_apple, normalized_apple_vec, normalized_snake_vec = self.game.get_angle_with_apple()
                direction, button_direction = self.game.generate_random_direction(angle_with_apple)
                _, is_front_blocked, is_left_blocked, is_right_blocked = self.game.blocked_directions()

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

    def run_game_with_model(self, model):
        max_score = 0
        avg_score = 0

        for _ in range(1, self.training_games+1):
            print(f'GAME {_}')
            self.game.reset_game()
            for _ in range(self.steps_per_game):
                angle_with_apple, normalized_apple_vec, normalized_snake_vec = self.game.get_angle_with_apple()
                direction, button_direction = self.game.generate_random_direction(angle_with_apple)
                current_direction_vec, is_front_blocked, is_left_blocked, is_right_blocked = self.game.blocked_directions()

                predicted_direction = np.argmax(
                    np.array(
                        model.predict(
                            np.array([
                                is_left_blocked, is_front_blocked, is_right_blocked,
                                normalized_apple_vec[0], normalized_snake_vec[0],
                                normalized_apple_vec[1], normalized_snake_vec[1]
                            ]).reshape(-1, 7)
                        )
                    )
                ) - 1

                new_direction = np.array(self.game.body[0]) - np.array(self.game.body[1])
                if predicted_direction == -1:
                    new_direction = np.array([new_direction[1], -new_direction[0]])
                if predicted_direction == 1:
                    new_direction = np.array([-new_direction[1], new_direction[0]])

                button_direction = self.game.generate_button_direction(new_direction.tolist())

                if self.game.meet_obstacle(current_direction_vec):
                    break

                self.game.play_game_by_ai(button_direction)
                max_score = max(max_score, self.game.score)

            avg_score += self.game.score

        return max_score, avg_score / self.training_games



def setup_generator(training_games, steps_per_game):
    sg = SnakeGameForTraining()
    sg.SPEED = 5000
    tdg = TrainingDataGenerator(sg, training_games, steps_per_game)
    return tdg


def main_generate_data():
    tdg = setup_generator(2, 100)
    tdg.generate_training_data()
    with open(tdg.model_name + '.in.json', 'w') as json_file:
        json.dump(tdg.training_data_x, json_file)
    with open(tdg.model_name + '.out.json', 'w') as json_file:
        json.dump(tdg.training_data_y, json_file)


def main_generate_model():
    tdg = setup_generator(2, 100)
    tdg.generate_training_data()

    model = Sequential()
    model.add(Dense(units=9, input_dim=7))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dense(output_dim=3,  activation = 'softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(
        (np.array(tdg.training_data_x).reshape(-1,7)),
        (np.array(tdg.training_data_y).reshape(-1,3)),
        batch_size = 256,
        epochs = 3
    )

    model.save_weights(tdg.model_name + '.h5')
    model_json = model.to_json()
    with open(tdg.model_name + '.json', 'w') as json_file:
        json_file.write(model_json)


def main_apply_model():
    tdg = setup_generator(2, 100)

    # load json model
    json_file = open(tdg.model_name + '.json', 'r')
    loaded_json_model = json_file.read()

    # load h5 weights
    model = model_from_json(loaded_json_model)
    model.load_weights(tdg.model_name + '.h5')

    tdg.run_game_with_model(model)
