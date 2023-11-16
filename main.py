from Experiment0 import Experiment0
from Experiment1 import Experiment1
from Experiment2 import Experiment2
from Experiment3 import Experiment3
from Experiment4 import Experiment4

# Ruta de la carpeta "train"
train_path = "a2/data/train"


def main():
    # Instanciar la clase y ejecutar el experimento
    experiment0 = Experiment0(train_path)
    experiment0.run_experiment()

    # experiment1 = Experiment1(train_path)
    # experiment1.run_experiment()
    #
    # experiment2 = Experiment2(train_path)
    # experiment2.run_experiment()
    #
    # experiment3 = Experiment3(train_path)
    # experiment3.run_experiment()
    #
    # experiment4 = Experiment4(train_path)
    # experiment4.run_experiment()


if __name__ == "__main__":
    main()
