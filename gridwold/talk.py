class Talk:

    def __init__(self, user_sentence, unibo_sentence):
        self.userSentence = user_sentence
        self.uniboSentence = unibo_sentence


if __name__ == "__main__":
    talk = Talk("abc", "aaa")
    print(type(talk))
    talk = Talk("abc", "aaa")
    print( type( talk ) )
