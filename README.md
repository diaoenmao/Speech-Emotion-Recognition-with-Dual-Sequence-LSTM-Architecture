# Speech Emotion Recognition with Dual-Sequence LSTM Architecture

[ICASSP 2020] This is an implementation of [Speech Emotion Recognition with Dual-Sequence LSTM Architecture](https://arxiv.org/abs/1910.08874) (DS-LSTM)
 - Dual-Sequence LSTM (DS-LSTM)
![ds-lstm](/img/ds-lstm.png)
 - Dual-level model with DS-LSTM cell
 ![diagram](/img/diagram.png)
## Requirements
 - Python 3
 - PyTorch 1.0

## Results
 - IEMOCAP  
 
|                                        |  WA  |  UA  |
|:--------------------------------------:|:----:|:----:|
|           M<sub>DS-LSTM</sub>          | 69.4 | 69.5 |
| M<sub>LSTM</sub> + M<sub>DS-LSTM</sub> | 72.7 | 73.3 |
## Acknowledgement
*Jianyou Wang  
Michael Xue  
Ryan Culhane  
Enmao Diao  
Jie Ding  
Vahid Tarokh*
