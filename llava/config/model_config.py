


# vanilla training
model_config_v0 = {
    "use_alternative": False,
    "projection_type": 'v0',
    "projector_loc": "after_vision_tower",
}


# matryoshka training
model_config_v4 = {
    "use_alternative": True,
    "projection_type": "v4",
    "matryoshka_vis_token_scale": None,
    "moe": None,
    "projector_loc": "after_vision_tower",
}