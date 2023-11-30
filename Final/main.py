from Exp7 import Exp7
from Exp8 import Exp8
from Exp9 import Exp9
from Test import Test

# Ruta de la carpeta "train"
train_path = "a2/data/train"
test_path = "a2/data/test"


def main():
    # Instanciar la clase y ejecutar el experimento
    # exp7 = Exp7(train_path)
    # exp7.run_experiment()

    # exp8 = Exp8(train_path)
    # exp8.run_experiment()
    #
    # exp9 = Exp9(train_path)
    # exp9.run_experiment()


    test = Test(train_path, test_path)
    test.run_experiment()


if __name__ == "__main__":
    main()
