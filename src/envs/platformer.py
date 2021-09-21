import pygame
import numpy as np

from src.envs.consts import *
from src.envs.envs_core import EnvBase
from src.envs.env_utils import run_env_with_display, HumanController, CoordinateTransformer

CELL_TYPE_BACKGROUND = 'background'
CELL_TYPE_GROUND = 'ground'

SCREEN_ABSOLUTE_WIDTH = 16
SCREEN_ABSOLUTE_HEIGHT = 12
CELL_SCREEN_HEIGHT = 1 / SCREEN_ABSOLUTE_HEIGHT
CELL_SCREEN_WIDTH = 1 / SCREEN_ABSOLUTE_WIDTH
START_CELL_INDEX = 0


def world_to_screen_coordinates(world_coordinates, camera_position):
    screen_coordinates = (
        (world_coordinates[0] - camera_position[0] + SCREEN_ABSOLUTE_WIDTH / 2) / SCREEN_ABSOLUTE_WIDTH,
        (world_coordinates[1] - camera_position[1] + SCREEN_ABSOLUTE_HEIGHT / 2) / SCREEN_ABSOLUTE_HEIGHT
    )
    return screen_coordinates


class Player:
    def __init__(self, world, move_speed=0.1):
        self.world = world
        self.position = [3.5, 10.1]
        self.world_grid_position = [np.floor(self.position[0]).astype(np.uint8),
                                    np.ceil(self.position[1]).astype(np.uint8)]
        self.move_speed = move_speed
        self.fall_acceleration = self.move_speed / 8
        self.screen_coordinates = (0.5, 0.5)
        self.color = COLORS_DICT['blue']
        self._falling = False
        self._jumping = False
        self._jumping_counter = 0
        self.velocity = [0.0, 0.0]

        self.width = 0.5
        self.height = 0.5

        self.draw_grid_ghost = False

        self.coordinate_transformer = None

    def update(self, action):
        done=False
        self.world.grid[self.world_grid_position[1], self.world_grid_position[0]].player_here = False
        self._falling = True
        ground_below = self.world.grid[
                           self.world_grid_position[1] - 1, self.world_grid_position[0]].cell_type == CELL_TYPE_GROUND
        if ground_below and self.position[1] % 1 < 2 * self.move_speed:
            self.velocity[1] = 0
            self.position[1] = self.world_grid_position[1] - 1
            self._falling = False

        if action == 1:
            self.position[0] -= self.move_speed
        elif action == 2:
            self.position[0] += self.move_speed
        elif action == 3:
            if not self._falling:
                self.velocity[1] += 0.25
                self._jumping = True

        # if self._jumping:
        #     self._jumping_counter += 1
        #     self.position[1] += self.move_speed
        #     if self._jumping_counter >= 50:
        #         self._jumping = False

        self.position[1] += self.velocity[1]
        self.velocity[1] -= self.fall_acceleration

        if self.world.grid[self.world_grid_position[1], self.world_grid_position[0] + 1].cell_type == CELL_TYPE_GROUND:
            self.position[0] = np.min([self.position[0], self.world_grid_position[0] + 1 - self.width / 2])
        if self.world.grid[self.world_grid_position[1], self.world_grid_position[0] - 1].cell_type == CELL_TYPE_GROUND:
            self.position[0] = np.max([self.position[0], self.world_grid_position[0] + self.width / 2])
        if self.world.grid[self.world_grid_position[1] + 1, self.world_grid_position[0]].cell_type == CELL_TYPE_GROUND:
            self.position[1] = np.min([self.position[1], self.world_grid_position[1] - self.height])

        if self.world_grid_position[0] - 1 == 0:
            self.position[0] = np.max([self.position[0], 1+self.width / 2])

        self.world_grid_position =np.array( [np.floor(self.position[0]),
                                    np.ceil(self.position[1] + 1e-8)])
        if self.world_grid_position[0] < 0 or self.world_grid_position[1] < 0:
            done = True
            return done
        self.world_grid_position = self.world_grid_position.astype(np.uint8)
        self.world.grid[self.world_grid_position[1], self.world_grid_position[0]].player_here = True

        return done

    def render(self, game_display, camera_position):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)

        self.screen_coordinates = world_to_screen_coordinates(self.position, camera_position)

        c2s = self.coordinate_transformer.cartesian_to_screen
        player_rect = pygame.Rect(*c2s(self.screen_coordinates),
                                  self.coordinate_transformer.get_screen_width(self.width) / SCREEN_ABSOLUTE_WIDTH + 1,
                                  self.coordinate_transformer.get_screen_height(
                                      self.height) / SCREEN_ABSOLUTE_HEIGHT + 1)
        player_rect.midbottom = c2s(self.screen_coordinates)
        pygame.draw.rect(game_display, self.color,
                         player_rect)

        pygame.draw.circle(game_display, COLORS_DICT['red'],
                           player_rect.midbottom, radius=5)

        if self.draw_grid_ghost:
            pygame.draw.rect(game_display, (0, 255, 0),
                             [*c2s(world_to_screen_coordinates(self.world_grid_position, camera_position)),
                              self.coordinate_transformer.get_screen_width(CELL_SCREEN_WIDTH) + 1,
                              self.coordinate_transformer.get_screen_height(CELL_SCREEN_HEIGHT) + 1], width=3)


class EnvCell:
    def __init__(self, coordinates=(0, 0), cell_type=CELL_TYPE_BACKGROUND):
        self.cell_type = cell_type
        self.coordinates = coordinates
        self.screen_coordinates = coordinates

        self.coordinate_transformer = None
        self.color = (100, 200, 255)
        self.player_here = False

    def set_type(self, cell_type):
        self.cell_type = cell_type

    def render(self, game_display, camera_position):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)

        self.screen_coordinates = world_to_screen_coordinates(self.coordinates, camera_position)
        if -CELL_SCREEN_WIDTH < self.screen_coordinates[0] < 1 + CELL_SCREEN_WIDTH and \
                -CELL_SCREEN_HEIGHT < self.screen_coordinates[1] < 1 + CELL_SCREEN_HEIGHT:
            display_width, display_height = pygame.display.get_surface().get_size()

            c2s = self.coordinate_transformer.cartesian_to_screen

            pygame.draw.rect(game_display, self.color,
                             [*c2s(self.screen_coordinates),
                              self.coordinate_transformer.get_screen_width(CELL_SCREEN_WIDTH) + 1,
                              self.coordinate_transformer.get_screen_height(CELL_SCREEN_HEIGHT) + 1])


class WorldGrid:
    def __init__(self, world_height, world_width):
        self._height = world_height
        self._width = world_width

        self.grid = np.zeros((self._height, self._width), dtype=EnvCell)

    def make_world(self):
        for row in range(self._height):
            for col in range(self._width):
                self.grid[row, col] = EnvCell(coordinates=(col, row))

        row = 10
        for col in range(self._width):
            if 10 < col < 12:
                continue
            self.grid[row, col].set_type(CELL_TYPE_GROUND)
            self.grid[row, col].color = (np.random.randint(100, 200), 100, 100)
        self.grid[11, 2].set_type(CELL_TYPE_GROUND)
        self.grid[11, 2].color = (np.random.randint(100, 200), 100, 100)
        self.grid[11, 4].set_type(CELL_TYPE_GROUND)
        self.grid[11, 4].color = (np.random.randint(100, 200), 100, 100)


class ScreenGrid:
    pass


class PlatformerEnv(EnvBase):
    def __init__(self, world_grid):
        super(PlatformerEnv, self).__init__()
        self.camera_position = [25.5, 10]
        self.world_grid = world_grid

        self.coordinate_transformer = None
        self.steps = 0
        self.player = Player(world_grid)

    def step(self, action):
        done = self.player.update(action)

        state = 1
        reward = 0
        return state, reward, done

    def reset(self):
        pass

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        display_width, display_height = pygame.display.get_surface().get_size()
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)

        # Camera behaviour at world borders
        self.camera_position[0] = max(self.player.position[0], SCREEN_ABSOLUTE_WIDTH / 2 + 1)
        self.camera_position[0] = min(self.camera_position[0], self.world_grid._width - SCREEN_ABSOLUTE_WIDTH / 2 - 1)
        self.camera_position[1] = max(self.player.position[1], SCREEN_ABSOLUTE_HEIGHT / 2 + 1)
        self.camera_position[1] = min(self.camera_position[1], self.world_grid._height - SCREEN_ABSOLUTE_HEIGHT / 2 - 1)

        game_display.fill(COLORS_DICT['black'])
        for grid_row in self.world_grid.grid:
            for grid_cell in grid_row:
                grid_cell.render(game_display, self.camera_position)
                # break
            # break
        self.player.render(game_display, self.camera_position)
        # pygame.draw.rect(game_display, COLORS_DICT['white'], [*c2s((0.1, 0.9)), *c2s((0.5, 0.5))])
        pygame.display.update()


if __name__ == '__main__':
    wg = WorldGrid(20, 40)
    wg.make_world()

    p_env = PlatformerEnv(wg)

    # key_map = {'K_LEFT': 1, 'K_RIGHT': 2}
    human_player = HumanController()
    agent = human_player.pick_action

    run_env_with_display(runs=1, env=p_env, agent=agent, frame_rate=60)
