#PPO_16/PPO_60
# axial_cfg = dict(
    #     num_blocks=2,              # 1–2 blocks are plenty here
    #     embed_dim=128,             # 64–128
    #     num_heads=4,               # 2–4
    #     positional_encoding="relative",  # distance-based
    #     feature_pool="mean",
    # )

    # policy_kwargs = dict(
    #     features_extractor_class=AxialTransformerFeaturesExtractor,
    #     features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
    #     net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    # ) #this config learnt fine: PPO_16

    # model = PPO(
    #         "MultiInputPolicy",
    #         env,                       # your VecEnv with TransformerObsWrapper
    #         policy_kwargs=policy_kwargs,
    #         n_steps=256, batch_size=256, n_epochs=20,
    #         #n_epochs=20, was 20 in first model
    #         learning_rate=3e-4,#slightly lower, so maybe 0.0001
    #         gamma=0.99, gae_lambda=0.95,
    #         clip_range=0.2, 
    #         clip_range_vf=10.0,
    #         vf_coef=0.25, ent_coef=0.02,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
    #         max_grad_norm=0.5,
    #         verbose=1,
    #         tensorboard_log='./output_malota'
    #     )

#PPO_19 - not sure what were the settings there

#PPO_61, 62 and 63, my best agent, learnt for 100 tys. steps
    # axial_cfg = dict(
    #     num_blocks=2,              # 1–2 blocks are plenty here
    #     embed_dim=64,             # 64–128
    #     num_heads=2,               # 2–4
    #     positional_encoding="relative",  # distance-based
    #     feature_pool="mean",
    # )

    # policy_kwargs = dict(
    #     features_extractor_class=AxialTransformerFeaturesExtractor,
    #     features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
    #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    # ) #this config learnt fine: PPO_16

    # model = PPO(
    #         "MultiInputPolicy",
    #         env,                       # your VecEnv with TransformerObsWrapper
    #         policy_kwargs=policy_kwargs,
    #         n_steps=256, batch_size=256, n_epochs=20,
    #         #n_epochs=20, was 20 in first model
    #         learning_rate=3e-4,#slightly lower, so maybe 0.0001
    #         gamma=0.99, gae_lambda=0.95,
    #         clip_range=0.2, 
    #         clip_range_vf=10.0,
    #         vf_coef=0.25, ent_coef=0.02,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
    #         max_grad_norm=0.5,
    #         verbose=1,
    #         tensorboard_log='./output_malota'
    #     )