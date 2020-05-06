import json

from keras.models import model_from_json, Sequential
from keras.layers import Dense
import numpy as np

from snake_deeplearning.game import SnakeGameForTraining
from snake_deeplearning.feed_forward_neural_network import forward_propagation
from snake_deeplearning.genetic_algorithm import GeneticAlgorithm


class TrainingDataGenerator:

    def __init__(self, game, training_games, steps_per_game):
        self.game = game
        self.training_data_x = []  # input
        self.training_data_y = []  # output
        self.training_games = training_games
        self.steps_per_game = steps_per_game
        self.model_name = f'model_{training_games}_{steps_per_game}'

    def generate_training_data(self):
        for game_id in range(1, self.training_games+1):
            print(f'GAME {game_id}')
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
                self.game.play_game_by_ai(button_direction, game_id)

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

        for game_id in range(1, self.training_games+1):
            print(f'GAME {game_id}')
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

                self.game.play_game_by_ai(button_direction, game_id)
                max_score = max(max_score, self.game.score)

            avg_score += self.game.score

        return max_score, avg_score / self.training_games

    def run_game_with_genetic_algorithm(self, weights, generation, chromosome, sol_per_pop):
        foods = 0
        expected_points = 0
        tmp_score = 0
        final_score = 0

        for game_id in range(1, self.training_games+1):
            if not generation:
                print(f'GAME {game_id}')
            self.game.reset_game()
            count_same_direction = 0
            prev_direction = 0
            for steps in range(1, self.steps_per_game+1):
                angle_with_apple, normalized_apple_vec, normalized_snake_vec = self.game.get_angle_with_apple()
                direction, button_direction = self.game.generate_random_direction(angle_with_apple)
                current_direction_vec, is_front_blocked, is_left_blocked, is_right_blocked = self.game.blocked_directions()

                predicted_direction = np.argmax(
                    np.array(
                        forward_propagation(
                            np.array([
                                is_left_blocked, is_front_blocked, is_right_blocked,
                                normalized_apple_vec[0], normalized_snake_vec[0],
                                normalized_apple_vec[1], normalized_snake_vec[1]
                            ]).reshape(-1, 7),
                            weights
                        )
                    )
                ) - 1

                if predicted_direction == prev_direction:
                    count_same_direction += 1
                else:
                    count_same_direction = 0
                    prev_direction = predicted_direction

                new_direction = np.array(self.game.body[0]) - np.array(self.game.body[1])
                if predicted_direction == -1:
                    new_direction = np.array([new_direction[1], -new_direction[0]])
                if predicted_direction == 1:
                    new_direction = np.array([-new_direction[1], new_direction[0]])

                button_direction = self.game.generate_button_direction(new_direction.tolist())

                if self.game.meet_obstacle(current_direction_vec):
                    # minus 30000 points if hitting obstacles
                    points = tmp_score + foods * 5000 - expected_points * 250 - 30000
                    break

                self.game.play_game_by_ai(button_direction, game_id, generation, chromosome, sol_per_pop, final_score)
                foods = max(foods, self.game.score)

                '''
                # give penalty if moving forward more than 8 steps
                if count_same_direction > 8 and predicted_direction != 0:
                    tmp_score -= 1
                else:
                    tmp_score += 2
                '''

                if steps % 100 == 0:
                    expected_points += 1

                final_score = tmp_score + foods * 5000 - expected_points * 250

        return final_score



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
    tdg = setup_generator(1000, 2000)

    # load json model
    json_file = open(tdg.model_name + '.json', 'r')
    loaded_json_model = json_file.read()

    # load h5 weights
    model = model_from_json(loaded_json_model)
    model.load_weights(tdg.model_name + '.h5')

    tdg.run_game_with_model(model)


def main_apply_genetic_algorithm():
    # training_games should always be 1 single it's controlled by sol_per_pop instead
    tdg = setup_generator(1, 2000)

    # step to next generation
    # (if the number is too low to find next population, the program will crash.)
    sol_per_pop = 50
    num_parents_mating = 12

    # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    n_x = 7  # no. of input units
    n_h = 9  # no. of units in hidden layer 1
    n_h2 = 15  # no. of units in hidden layer 2
    n_y = 3  # no. of output units
    num_weights = n_x * n_h + n_h * n_h2 + n_h2 * n_y

    # Defining the population size.
    pop_size = (sol_per_pop, num_weights)
    # Creating the initial population.
    new_population = np.random.choice(np.arange(-1, 1, step=0.01), size=pop_size, replace=True)

    # Loading previous population
    # with open('models/model_1_2000.generation1000.json', 'r') as f:
    #    new_population = np.array(json.load(f))

    num_generations = 1000
    for generation in range(1, num_generations+1):
        print(f'generation {generation}')
        # Measuring the fitness of each chromosome in the population.
        fitness = GeneticAlgorithm.cal_pop_fitness(
            tdg.run_game_with_genetic_algorithm, new_population,
            generation, sol_per_pop
        )
        print(f'fittest chromosome fitness value: {np.max(fitness)}')
        # Selecting the best parents in the population for mating.
        parents = GeneticAlgorithm.select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = GeneticAlgorithm.crossover(
            parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights)
        )

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = GeneticAlgorithm.mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # save the training result per 100 rounds
        # if generation % 100 == 0:
        #    with open(tdg.model_name + f'.generation{generation}.json', 'w') as json_file:
        #        json.dump(new_population.tolist(), json_file)
