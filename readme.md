# Unet-DAE
A computational model of Mesgarani-12

## Procedure of running this program
- scp .wav files (plz contact the author)

- Prepare dataset
  - You could do this by running command:

    ```shell
    python3 __init__.py
    ```

    - This converts .wav files to .json files. This gets a pool of .json files that represent speech segments of less than second. Each .json file contain a 2-d matrix, which is the spectrogram of the incoming speech.

- Training

  - Run training.py after you have populated directories 'trainset' and 'testset'. Run command:

    ```shell
    python3 training.py
    ```

  - Running the training takes three to four hours.

- Testing

  - Run test.py by typing the following command:

    ```shell
    python3 test.py
    ```

  - Results can be seen in directory 'result'. The result of a single test is capsulated in directories under 'result', named as 'testid=x_y', where 'x' and 'y' are indexes (used just to distinguish different files).

  - In every 'testid=x_y' file there are 7 pngs: 

    - xxx_clean_0.png: raw spectrogram of A
    - xxx_clean_1.png: raw spectrogram of B
    - xxx_mixed.png: mixed
    - xxx_recover_atten=0: recovery from mixed when attending A
    - xxx_recover_atten=1: recovery from mixed when attending B
    - xxx_single_0: recovery from A's raw spectrogram
    - xxx_single_1: recovery from B's raw spectrogram