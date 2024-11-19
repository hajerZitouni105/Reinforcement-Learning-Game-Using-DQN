from keras.models import load_model

# EX : say you have the model in a folder called "models" and model's name is "myModel.model"
model_name_ = "model.keras"
model = load_model(f'models/{model_name_}')


# Now We can use the trained agent to play the game :
# Next code initializes the environment and the agent 
# then uses the trained agent to play 20 rounds of the game and 
# records the score and time of each round 
env2 = Env()
env2.WALL_SPEED = 1

start = time.time()
for _ in range(20):
    WINDOW          = pygame.display.set_mode((env2.WINDOW_WIDTH, env2.WINDOW_HEIGHT))
    clock           = pygame.time.Clock()
    score_increased = False
    game_over       = False
    _ = env2.reset()
    pygame.display.set_caption("Game")
    while not game_over:
        clock.tick(27)
        prd = model.predict((env2.field.body/env2.MAX_VAL).reshape(-1, *env2.ENVIRONMENT_SHAPE))
        actions = [np.argmax(prd[0]),np.argmax(prd[0])]
        _,reward, game_over = env2.step(actions)
        env2.render(WINDOW = WINDOW)


    #####################################################
    a = int(time.time()-start)
    print(f"Score {env2.score} in {a//60}:{a%60}")
    sleep(0.5)
    WINDOW.fill(env2.WHITE)
    env2.print_text(WINDOW = WINDOW, text_cords = (env2.WINDOW_WIDTH // 2, env2.WINDOW_HEIGHT// 2),
                       text = f"Game Over - Score : {env2.score}", color = env2.RED, center = True)
    pygame.display.update()
    sleep(0.1)
pygame.quit()