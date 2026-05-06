from src.predict import generate_caption

def main():
    # change this path to any image you want
    image_path = "data/test.jpg"

    print("Processing image...")

    caption = generate_caption(image_path)

    print("\nGenerated Caption:")
    print(caption)


if __name__ == "__main__":
    main()