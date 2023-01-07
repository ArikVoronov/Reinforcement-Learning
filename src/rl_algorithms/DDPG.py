import copy
import torch
from pytorch_model_summary import summary
from src.utils.models import DenseContinuousActorModel, DenseContinuousQModel


class AlgorithmDDPG:
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
                 reward_discount, target_update_interval, polyak):
        self.env = env

        # Q Approximation Model
        self._target_update_interval = target_update_interval
        self._polyak = polyak
        self._model_learning_rate = model_learning_rate
        self._create_model(model_config)

        # RL Parameters
        self.reward_discount = reward_discount

    def _create_model(self, model_config):
        # Create actor model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_limit = self.env.action_space.high[0]
        actor_model = DenseContinuousActorModel(input_size=state_dim,
                                                n_actions=act_dim,
                                                hidden_size_list=model_config['hidden_layers_dims'],
                                                act_limit=act_limit,
                                                device=self._device)

        self.actor_model = actor_model.to(self._device)
        self.actor_model_target = copy.deepcopy(self.actor_model)

        self._actor_optimizer = torch.optim.RMSprop(self.actor_model.parameters(), lr=self._model_learning_rate)

        # Create Q model
        # Input of the shape [state_dim+action_dim] output q value
        q_model = DenseContinuousQModel(state_dim=state_dim,
                                        n_actions=act_dim,
                                        hidden_size_list=model_config['hidden_layers_dims'],
                                        device=self._device)
        self.q_model = q_model.to(self._device)
        self._q_optimizer = torch.optim.RMSprop(self.q_model.parameters(), lr=self._model_learning_rate)

        self.q_model_target = copy.deepcopy(self.q_model)

    def optimize_step(self, data_batch):
        state_batch = torch.cat(data_batch.state).to(self._device)
        action_batch = torch.cat(data_batch.action).to(self._device)
        reward_batch = torch.cat(data_batch.reward).to(self._device)
        batch_size = state_batch.shape[0]

        # Q model optimization
        self._q_optimizer.zero_grad()
        x = torch.cat([state_batch, action_batch], dim=-1)
        q = self.q_model(x)  # current after next to save forward context

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                data_batch.next_state)), device=self.q_model.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in data_batch.next_state
                                           if s is not None]).to(self.q_model.device)
        q_next = torch.zeros([batch_size, q.shape[1]], device=self.q_model.device)

        with torch.no_grad():
            next_actions = self.actor_model(non_final_next_states)
            x = torch.cat([non_final_next_states, next_actions], dim=-1)
            q_next[non_final_mask] = self.q_model(x)
            q_next = q_next.detach()
            targets = reward_batch.unsqueeze(1) + self.reward_discount * q_next

        q_loss = torch.mean((targets - q) ** 2)
        q_loss.backward()
        self._q_optimizer.step()

        # Actor optimization
        self._actor_optimizer.zero_grad()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.q_model.parameters():
            p.requires_grad = False
        predicted_actions_batch = self.actor_model(state_batch)
        x = torch.cat([state_batch, predicted_actions_batch], dim=-1)
        actor_loss = torch.mean(-self.q_model(x))
        actor_loss.backward()
        self._actor_optimizer.step()
        for p in self.q_model.parameters():
            p.requires_grad = True

    def pick_action(self, state):
        with torch.no_grad():
            if not type(state) == torch.Tensor:
                state = torch.tensor(state, device=self._device)
            chosen_action = self.actor_model(state)
        return chosen_action

    def pick_test_action(self, state):
        with torch.no_grad():
            if not type(state) == torch.Tensor:
                state = torch.tensor(state, device=self._device)
            chosen_action = self.actor_model(state)
        return chosen_action.item()

    def on_episode_end(self, episode):
        if (self._target_update_interval > 0) and (episode % self._target_update_interval == 0):
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.q_model_target.parameters(), self.q_model_target.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self._polyak)
                    p_targ.data.add_((1 - self._polyak) * p.data)
