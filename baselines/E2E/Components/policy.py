import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .utils.distribution import Categorical


class GoalAttentionModule(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, goal_embed, down_rgb_tokens):
        q = goal_embed.unsqueeze(1)        # [B, 1, D]
        out, _ = self.attn(q, down_rgb_tokens, down_rgb_tokens)  # [B, 1, D]
        return out.squeeze(1)              # [B, D]



class GridEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # 60x60 → 30x30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 30x30 → 15x15
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 15 * 15, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)




class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs
    




class Explore_Network(NNBase):
    def __init__(self,
                 rgb_embed_dim=2048,
                 grid_embed_dim=128,
                 action_embed_dim=32,
                 position_embed_dim=32,  # includes altitude
                 goal_embed_dim=128,
                 time_embed_dim=16,
                 down_rgb_patch_dim=256,
                 fusion_dim=512,
                 rnn_hidden_size=512,
                 action_dim=6):
        super().__init__()

        # super(Explore_Network, self).__init__(
        #     recurrent, hidden_size, hidden_size)

        self.grid_encoder = GridEncoder(input_channels=1, output_dim=grid_embed_dim)

        self.rgb_proj = nn.Linear(rgb_embed_dim, fusion_dim)
        self.grid_proj = nn.Linear(grid_embed_dim, fusion_dim)
        self.act_embed = nn.Embedding(action_dim, action_embed_dim)
        self.act_proj = nn.Linear(action_embed_dim, fusion_dim)
        self.pos_proj = nn.Linear(position_embed_dim, fusion_dim)
        self.goal_proj = nn.Linear(goal_embed_dim, fusion_dim)
        self.time_proj = nn.Linear(time_embed_dim, fusion_dim)
        self.down_proj = nn.Linear(down_rgb_patch_dim, fusion_dim)

        self.attn = GoalAttentionModule(embed_dim=fusion_dim)

        self.gru = nn.GRU(fusion_dim * 7, rnn_hidden_size, batch_first=True)

        self.critic = nn.Linear(rnn_hidden_size, 1)
        self.actor = nn.Linear(rnn_hidden_size, action_dim)

    def forward(self, 
                rgb_embed,
                grid_map,
                prev_action,
                pos_embed,
                goal_embed,
                time_embed,
                down_rgb_tokens,
                rnn_hxs):

        B = rgb_embed.size(0)
        grid_map = grid_map.unsqueeze(1)  # [B, 1, H, W]
        grid_embed = self.grid_encoder(grid_map)

        rgb_feat = self.rgb_proj(rgb_embed)
        grid_feat = self.grid_proj(grid_embed)
        act_feat = self.act_proj(self.act_embed(prev_action))
        pos_feat = self.pos_proj(pos_embed)  # includes altitude
        goal_feat = self.goal_proj(goal_embed)
        time_feat = self.time_proj(time_embed)
        down_attn = self.attn(goal_feat, self.down_proj(down_rgb_tokens))  # [B, D]

        x = torch.cat([
            rgb_feat, grid_feat, act_feat, pos_feat,
            goal_feat, time_feat, down_attn
        ], dim=-1)  # [B, fusion_dim * 7]

        x, rnn_hxs = self.gru(x.unsqueeze(1), rnn_hxs)  # [B, 1, H]
        x = x.squeeze(1)

        return self.actor(x), self.critic(x), rnn_hxs
    



class SimpleActorCritic(nn.Module):
    # def __init__(self, obs_dim, hidden_size=128):
    def __init__(self, obs_space, hidden_size=128):
        super().__init__()


        if isinstance(obs_space, gym.spaces.Dict):
            obs_dim = {}
            for key, space in obs_space.spaces.items():
                obs_dim[key] = space.shape[0]
        else:
            obs_dim = obs_space.shape[0]

        self.obs_dim = obs_dim

        # print('obs_dim:', obs_dim)

        self.output_size = hidden_size

        # actor -----------------------------------------------------------
        # pos
        self.fc_pos = nn.Sequential(
            nn.Linear(obs_dim['pos'], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # explored map
        self.map_conv = nn.Sequential(
            nn.Conv2d(obs_dim['map'], 16, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H/8
            nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d(1)                                # -> (B,64,1,1)
            nn.AdaptiveAvgPool2d(output_size=3),  # 输出 3×3 coarse patch
            nn.Flatten()
        )
        # self.fc_map = nn.Linear(64, hidden_size)
        self.fc_map = nn.Linear(64 * 3 * 3, hidden_size)


        # linear layer for combining features
        # self.linear1 = nn.Linear(2*hidden_size, 2*hidden_size)

        # actor and critic -----------------------------------------------------------
        self.actor = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
          )  
        
        self.critic = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.is_recurrent = False

        # self._init_weights()


    def _init_weights(self):
        """Orthogonal init (IMPALA style) + 差异化 actor / critic 头"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # 细调输出层
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)   # 小增益→初始 logits≈0
        nn.init.constant_(self.actor[-1].bias, 0.0)

        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)   # 预测值接近 0
        nn.init.constant_(self.critic[-1].bias, 0.0)

        

    def forward(self, obs):
        # x = self.fc(obs)
        ## self.critic_linear(x).squeeze(-1), x, rnn_hxs
        # return self.critic(x).squeeze(-1), self.actor(x)

        pos = obs['pos']  # [B, pos_dim]
        grid_map = obs['map']  
        
        x_pos = self.fc_pos(pos)
        x_map = self.map_conv(grid_map)
        x_map = x_map.view(x_map.size(0), -1)  # flatten to [B, hidden_size]
        x_map = self.fc_map(x_map)  # [B, hidden_size]
        # x_map = self.fc_map(grid_map)  # [B, hidden_size]
        x = torch.cat((x_pos, x_map), 1)
        # x = nn.ReLU()(self.linear1(x))

        return self.critic(x).squeeze(-1), self.actor(x)


    # # def evaluate_actions(self, obs_map, rec_states, masks, actions, extras=None):
    # def evaluate_actions(self, obs_map, actions):
    #     actor_out, critic_out = self.forward(obs_map)
    #     dist = torch.distributions.Categorical(logits=actor_out)
    #     action_log_probs = dist.log_prob(actions)
    #     dist_entropy = dist.entropy().mean()
    #     return critic_out.squeeze(-1), action_log_probs, dist_entropy, None
    


class SimpleActorCritic_obstacle(SimpleActorCritic):
    def __init__(self, obs_space, hidden_size=128):
        super().__init__(obs_space, hidden_size)

        self.obs_space = obs_space

        if len(obs_space['depth'].shape) < 3:
            ## vector
            self.depth_encoder = nn.Sequential(
                nn.Linear(self.obs_dim['depth'], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            depth_out_dim = hidden_size

        # elif obs_space['depth'].shape[-1] == 64:
        else:
            ## image
            # self.depth_encoder = nn.Sequential(
            #     nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> 32x32
            #     nn.ReLU(),
            #     nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 16x16
            #     nn.ReLU(),
            #     nn.Flatten(),
            # )
            # depth_out_dim = 32 * 16 * 16
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 64→32
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32→16
                nn.ReLU(),
                nn.AvgPool2d(2),                            # 16→8  ← 池化
                nn.Flatten(),
            )
            depth_out_dim = 32 * 8 * 8      # = 2048
        
        # else:  # airsim depth image  (640 -> 64)
        #     self.depth_encoder = nn.Sequential(
        #         nn.AdaptiveAvgPool2d((64, 64)),                            # 64→6
        #         nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 64→32
        #         nn.ReLU(),
        #         nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32→16
        #         nn.ReLU(),
        #         nn.AvgPool2d(2),                            # 16→8  ← 池化
        #         nn.Flatten(),
        #     )
        #     depth_out_dim = 32 * 8 * 8      # = 2048


        
       # region
        # self.actor = nn.Sequential(
        #     nn.Linear(3*hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size)
        #   )  
        
        # self.critic = nn.Sequential(
        #     nn.Linear(3*hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )
        # endregion

        self.merge_hidden_size = depth_out_dim + 2*hidden_size
        
        # merge all
        self.fc_merge = nn.Sequential(
            nn.Linear(self.merge_hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size)
          )  
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        
        
    def forward(self, obs):
        pos = obs['pos']  # [B, pos_dim]
        grid_map = obs['map']  
        
        x_pos = self.fc_pos(pos)
        x_map = self.map_conv(grid_map)
        x_map = x_map.view(x_map.size(0), -1)  # flatten to [B, hidden_size]
        x_map = self.fc_map(x_map)  # [B, hidden_size]
        # x_map = self.fc_map(grid_map)  # [B, hidden_size]

        x_depth = obs['depth']  # [B, depth_dim]
        if self.obs_space['depth'].shape[-1] != 64 and len(self.obs_space['depth'].shape) >= 3:
            x_depth = F.adaptive_avg_pool2d(x_depth, (64, 64))

        x_depth = self.depth_encoder(x_depth)  # [B, hidden_size]

        x = torch.cat((x_pos, x_map, x_depth), 1)

        x = self.fc_merge(x)  ## 新添加的这条代码
        # x = nn.ReLU()(self.linear1(x))

        return self.critic(x).squeeze(-1), self.actor(x)
    


class ActorCritic_obstacle_3Dfly(SimpleActorCritic_obstacle):
    def __init__(self, obs_space, hidden_size=128):
        super().__init__(obs_space, hidden_size)

        scaler_hidden_size = int(0.25 * hidden_size)

        self.height_encoder = nn.Sequential(
            nn.Linear(1, scaler_hidden_size),
            nn.ReLU()
        )
        self.distance_agent2objdown = nn.Sequential(
            nn.Linear(1, scaler_hidden_size),
            nn.ReLU()
        )

        
        merge_hidden_size = self.merge_hidden_size + 2*scaler_hidden_size
        self.fc_merge = nn.Sequential(
            nn.Linear(merge_hidden_size, hidden_size),
            nn.ReLU()
        )


    def forward(self, obs):
        pos = obs['pos']  # [B, pos_dim]
        grid_map = obs['map']  
        
        x_pos = self.fc_pos(pos)
        x_map = self.map_conv(grid_map)
        x_map = x_map.view(x_map.size(0), -1)  # flatten to [B, hidden_size]
        x_map = self.fc_map(x_map)  # [B, hidden_size]
        # x_map = self.fc_map(grid_map)  # [B, hidden_size]

        x_depth = obs['depth']  # [B, depth_dim]
        if self.obs_space['depth'].shape[-1] != 64 and len(self.obs_space['depth'].shape) >= 3:
            x_depth = F.adaptive_avg_pool2d(x_depth, (64, 64))

        x_depth = self.depth_encoder(x_depth)  # [B, hidden_size]

        ## height related
        x_agent_height = self.height_encoder(obs['agent_height'])
        x_distance = self.distance_agent2objdown(obs['distance_agent2objdown'])


        x = torch.cat((x_pos, x_map, x_depth, x_agent_height, x_distance), 1)
        x = self.fc_merge(x)  ## 新添加的这条代码
        # x = nn.ReLU()(self.linear1(x))

        return self.critic(x).squeeze(-1), self.actor(x)





class RL_Explore_Policy(nn.Module):

    # def __init__(self, obs_map_shape, obs_points_shape, action_space, model_type=0,
    #              base_kwargs=None):
    def __init__(self, obs_space, action_space, args=None,
                 base_kwargs=None):

        super(RL_Explore_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        model_type = args.env_mode

        if model_type == 'dummy':
            # self.network = Explore_Network(
            #     obs_map_shape, obs_points_shape, **base_kwargs)

            # self.network = SimpleActorCritic(
            #     obs_map_shape[0], **base_kwargs)
            self.network = SimpleActorCritic(
                obs_space, **base_kwargs)
        
        elif model_type == 'dummy_obstacle':
            self.network = SimpleActorCritic_obstacle(
                obs_space, **base_kwargs)
            
        elif model_type == 'dummy_fly_vertical':
            self.network = ActorCritic_obstacle_3Dfly(
                obs_space, **base_kwargs)
            
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    def load_pretrained_stage(self, stage, model_dir="./models/obstacle_3D"):
        """
        Load pretrained model from previous curriculum learning stage
        Args:
            stage: Previous stage number to load (current_stage - 1)
            model_dir: Directory containing saved models
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        import os
        import torch
        
        stage_dir = os.path.join(model_dir, f"stage_{stage}")
        model_path = os.path.join(stage_dir, "best_model.pth")
        
        if os.path.exists(model_path):
            try:
                print(f"Loading pretrained model from stage {stage}: {model_path}")
                state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
                self.load_state_dict(state_dict)
                print(f"Successfully loaded pretrained model from stage {stage}")
                return True
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                return False
        else:
            print(f"Pretrained model not found: {model_path}")
            return False

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    # def forward(self, inputs_map, inputs_points, rnn_hxs, masks, extras):
    #     if extras is None:
    #         return self.network(inputs_map, inputs_points , rnn_hxs, masks)
    #     else:
    #         return self.network(inputs_map, inputs_points, rnn_hxs, masks, extras)

    # def forward(self, inputs_map, rnn_hxs, masks, extras):
    def forward(self, obs, extras=None):
        # return self.network(inputs_map)
        if extras is None:
            return self.network(obs)
        else:
            return self.network(obs, extras)
        # if extras is None:
            # return self.network(inputs_map, rnn_hxs, masks)
        # else:
        #     return self.network(inputs_map, rnn_hxs, masks, extras)
        

    # def act(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None, deterministic=False):
    def act(self, obs, deterministic=False):

        # value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
        value, actor_features = self(obs)
        dist = self.dist(actor_features)

        # logits = dist.logits
        # probs = F.softmax(logits, dim=-1)

        # if torch.isnan(dist.logits).any():
        #     print('has nan in logits')
        # if torch.isinf(dist.logits).any():
        #     print('has inf in logits')


        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs
    

    # def act(self, obs, deterministic=False):
    #     value, actor_features, rnn_hxs 

    # def get_value(self, inputs_map, inputs_points, rnn_hxs, masks, extras=None):
    #     value, _, _ = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
    #     return value
    
    def get_value(self, obs):
        value, _ = self(obs)
        return value

    # def evaluate_actions(self, inputs_map, inputs_points, rnn_hxs, masks, action, extras=None):

    #     value, actor_features, rnn_hxs = self(inputs_map, inputs_points, rnn_hxs, masks, extras)
    #     dist = self.dist(actor_features)

    #     action_log_probs = dist.log_probs(action)
    #     dist_entropy = dist.entropy().mean()

    #     return value, action_log_probs, dist_entropy, rnn_hxss

    def evaluate_actions(self, obs, action):

        value, actor_features = self(obs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
    





if __name__ == "__main__":
    B, N = 2, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Explore_Network().to(device)

    rgb_embed = torch.randn(B, 2048).to(device)
    grid_map = torch.randn(B, 60, 60).to(device)
    prev_action = torch.randint(0, 7, (B,)).to(device)
    pos_embed = torch.randn(B, 32).to(device)  # includes altitude
    goal_embed = torch.randn(B, 128).to(device)
    time_embed = torch.randn(B, 16).to(device)
    down_rgb_tokens = torch.randn(B, N, 256).to(device)
    rnn_hxs = torch.zeros(1, B, 512).to(device)

    act_logits, value, rnn_hxs_out = net(
        rgb_embed, grid_map, prev_action,
        pos_embed, goal_embed, time_embed,
        down_rgb_tokens, rnn_hxs
    )

    print("Action logits:", act_logits.shape)
    print("Value:", value.shape)
    print("New RNN state:", rnn_hxs_out.shape)