import copy
import torch
from pytorch_model_summary import summary
from src.utils.models import DenseActorCriticModel


class AlgorithmActorCritic:
    """
    Actor Critic Algorithm

    Learn the value function explicitly optimal policy

    The value function is used for the optimization of the policy

    V(s) = state value
    Target = reward_discount*V(next_state)+reward(current_state)

    Delta = Target - V(s)

    Critic (value) loss = Delta**2 (so the derivative is delta exactly)
    Actor (policy) loss = -log(policy[current_action]) * Delta

    Action is selected by sampling from the policy distribution for the current state
    """

    def __init__(self, model_config, env, model_learning_rate,
                 reward_discount):
        self.env = env

        # Q Approximation Model
        self._model_learning_rate = model_learning_rate
        self._create_model(model_config)

        # RL Parameters
        self.reward_discount = reward_discount

    def _create_model(self, model_config):
        # Create model
        nn_model = DenseActorCriticModel(input_size=self.env.observation_space.shape[0],
                                         n_actions=self.env.action_space.n,
                                         hidden_size_list=model_config['hidden_layers_dims'])
        self._device = nn_model.device
        self.nn_model = nn_model.to(self._device)
        self._optimizer = torch.optim.RMSprop(self.nn_model.parameters(), lr=self._model_learning_rate)

        print(summary(self.nn_model, torch.zeros([1, self.env.observation_space.shape[0]]).to(self._device),
                      show_input=True))

    def load_weights(self, weights_file_path):
        self.nn_model.load_parameters_from_file(weights_file_path)

    def optimize_step(self, data_batch):
        state_batch = torch.cat(data_batch.state).to(self._device)
        action_batch = torch.cat(data_batch.action).to(self._device)
        reward_batch = torch.cat(data_batch.reward).to(self._device)
        batch_size = state_batch.shape[0]

        # Forward pass
        policy, value = self.nn_model(state_batch)  # current after next to save forward context

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                data_batch.next_state)), device=self.nn_model.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in data_batch.next_state
                                           if s is not None])
        value_next = torch.zeros([batch_size, value.shape[1]], device=self.nn_model.device)
        _, value_next[non_final_mask] = self.nn_model(non_final_next_states)
        value_next = value_next.detach()

        targets = reward_batch.unsqueeze(1) + self.reward_discount * value_next

        delta = targets - value

        # Train model
        self._optimizer.zero_grad()
        critic_loss = delta ** 2

        # actor_loss = torch.zeros(size=policy.shape, device=self._device)
        actor_loss = -torch.log(policy).gather(1, action_batch) * delta
        total_loss = torch.mean(actor_loss + critic_loss)
        total_loss.backward()
        self._optimizer.step()

    def pick_action(self, state):
        with torch.no_grad():
            if not type(state) == torch.Tensor:
                state = torch.tensor(state, device=self._device)
            policy, _ = self.nn_model(state)
            policy = torch.abs(policy)
            chosen_action = torch.multinomial(policy, policy.shape[0], replacement=True)
        return chosen_action

    def pick_test_action(self, state):
        with torch.no_grad():
            if not type(state) == torch.Tensor:
                state = torch.tensor(state, device=self._device)
            policy, _ = self.nn_model(state)
            chosen_action = torch.max(policy)
        return chosen_action.item()

    def on_episode_end(self, episode):
        pass
