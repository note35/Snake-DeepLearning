# Snake-DeepLearning

Applying Deep Learning on Snake Game


# Installation

PyGame supports up to python3.7; therefore, this project will stay at Python3.7. Below guide are tested in MacOS 10.14.6 only.


1. Install Poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.7
```

You can find your python path (Virtualenv -> Path) by running below command `python3.7 ~/.poetry/bin/poetry env info`.

```
Virtualenv
Python:         3.7.4
Implementation: CPython
Path:           /Users/xxx/Library/Caches/pypoetry/virtualenvs/snake-deeplearning-LtzCkVy--py3.7
Valid:          True

System
Platform: darwin
OS:       posix
Python:   /usr/local/Cellar/python/3.7.4_1/Frameworks/Python.framework/Versions/3.7
```

2. Install PyGame

Since PyGame is problematic for MacOS if you install it through regular approach, we need to comile it from source code. Below commands follow the guide from [StackOverflow](https://stackoverflow.com/a/59060598/2740386).

```bash
brew install sdl2 sdl2_gfx sdl2_image sdl2_mixer sdl2_net sdl2_ttf
git clone -b 1.9.6 https://github.com/pygame/pygame.git --single-branch
cd pygame
<Virtualenv python3.7 PATH>/bin/python3.7 setup.py -config -auto -sdl2
<Virtualenv python3.7 PATH>/bin/python3.7 setup.py install
cd ../
```

3. Install Project

```bash
python3.7 ~/.poetry/bin/poetry install
```

4. Run Game

```bash
python3.7 ~/.poetry/bin/poetry run game
```

5. Generate Training Model

```bash
python3.7 ~/.poetry/bin/poetry run generate_model
```

- Run 100 games with 2000 steps will get about 95% accurate result.
- This training model has one defect, it will hit self while head is surrending by body but apple appears in the other side of body. (Therefore, the model based on this training data will hit similar error.)
- This training model doesn't support additional obstacles in the map. One TODO work is to support this use case.

6. Run Game with Model

```bash
python3.7 ~/.poetry/bin/poetry run test_model
```

# Cleanup

```bash
python3.7 ~/.poetry/bin/poetry env remove 3.7
```

# Reference

- https://theailearner.com/2018/04/19/snake-game-with-deep-learning/
- https://github.com/TheAILearner/Snake-Game-with-Deep-learning
- https://github.com/TheAILearner/Training-Snake-Game-With-Genetic-Algorithm
