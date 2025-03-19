import nltk

def main():
    # print("Downloading required NLTK data...")
    # nltk.download("movie_reviews")
    # print("Download complete!")
    print("Downloading required NLTK data: wordnet...")
    nltk.download("wordnet")
    print("Download wordnet complete!")

if __name__ == "__main__":
    main()
