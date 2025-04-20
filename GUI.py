from utils import *
import tensorflow as tf
from MODEL.model import *
def main():
    # ===================== Preparation =====================
    # Load the MNIST dataset from local Keras storage
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and reshape the images to [num_samples, 28, 28, 1]
    x_train, x_test = imageNormalization(x_train, x_test, mean=0.1037, std=0.3081)

    # Convert the labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)  # 10 classes
    y_test = tf.keras.utils.to_categorical(y_test, 10)    # 10 classes

    print("x_train shape:", x_train.shape)  # (60000, 28, 28, 1)
    print("x_test shape:", x_test.shape)    # (10000, 28, 28, 1)
    print("y_train shape:", y_train.shape)  # (60000, 10)
    print("y_test shape:", y_test.shape)    # (10000, 10)

    # ===================== Train / Load Model =====================
    # model_path = "MODEL/cnn_model.h5"

    # Initialize the CNN model with input shape and model path
    # model = CNNModel(x_train.shape[1:], model_path=model_path)
    model = CapsuleModel(x_train.shape[1:])
    model.summary()

    # If model is not loaded, compile and train it
    if not model.is_load:
        model.train(x_train,y_train,epochs=20,batch_size=32)
        model.save()  # Save the trained model

    NDOCNNacc = model.evaluate(x_test,y_test,save_path=f"result/{model.label}_acc.csv")

    # ===================== Pygame Event Loop =====================
    draw_on = False  # Initialize draw_on variable before the loop
    try:
        while True:
            # Get all events
            e = pygame.event.wait()

            # Draw partition lines (function should be defined elsewhere)
            draw_partition_line()

            # Clear screen on right-click (button 3)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                screen.fill(white)

            # Quit the application on quit event
            if e.type == pygame.QUIT:
                raise StopIteration

            # Start drawing on left-click (button 1)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button != 3:
                color = black
                draw_on = True

            # Stop drawing after releasing left-click
            if e.type == pygame.MOUSEBUTTONUP and e.button != 3:
                draw_on = False
                fname = "ASSETS/out.png"

                # Crop the screen to an image
                img = crope(screen)
                pygame.image.save(img, fname)

                # Get prediction output and display results
                output_img, pred = get_output_image(fname, model)
                show_combined_output(output_img, pred)

            # Draw line on screen while moving the mouse if drawing is active
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos, radius)
                last_pos = e.pos

            # Update the screen display
            pygame.display.flip()

    except StopIteration:
        pass

    # Quit pygame after the loop ends
    pygame.quit()


# Entry point of the script, checking if this script is being run directly
if __name__ == "__main__":
    main()
