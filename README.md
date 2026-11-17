# brume

A denoising diffusion model (DDPM) for 2D point clouds, written in plain NumPy. No torch, no autograd library - the noise-prediction network, its backward pass, Adam, EMA, the noise schedule and both samplers (ancestral DDPM and DDIM) are all hand-written and small enough to read in one sitting.

I wanted to actually understand diffusion rather than just call `diffusers`, and 2D data is perfect for that: you can see immediately whether the model learned the shape or just a blurry blob, and training takes under a minute on a laptop CPU.

*brume* is French for mist - which is roughly what the model starts from.

## what's in here

- `schedule.py` - cosine schedule (Nichol & Dhariwal) and a rescaled linear one, plus the posterior variances
- `nn.py` - MLP noise predictor: sinusoidal timestep embedding re-injected into every hidden layer, Fourier features on the input coordinates, SiLU, manual backprop, Adam with global-norm clipping
- `diffusion.py` - forward process, epsilon-prediction loss, ancestral sampling, DDIM (deterministic with `eta=0`)
- `train.py` - training loop with warmup + cosine LR and an EMA of the weights
- `metrics.py` - sliced Wasserstein distance and multi-bandwidth RBF MMD to score samples
- `plot.py` - ascii density maps, because I wanted the whole thing to work over ssh

Datasets: `moons`, `spiral`, `rings`, `checker`, `gmm8` (all standardised to unit scale).

## running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m brume train --data spiral --show-steps --save spiral.npz
python -m brume sample spiral.npz --ddim 25 --data spiral
```

## a real run

Default settings (6000 steps, batch 512, T=200, 3x128 MLP), on CPU:

```
training on 'spiral' for 6000 steps (T=200)
step    750  loss 0.4630  lr 1.93e-03    5.1s
step   1500  loss 0.4480  lr 1.72e-03    9.9s
step   2250  loss 0.4498  lr 1.41e-03   14.7s
step   3000  loss 0.4440  lr 1.05e-03   19.7s
step   3750  loss 0.4458  lr 6.86e-04   24.6s
step   4500  loss 0.4441  lr 3.78e-04   30.0s
step   5250  loss 0.4332  lr 1.72e-04   34.5s
step   6000  loss 0.4412  lr 1.00e-04   38.8s

                      data                        |                   brume samples                  
                                                  |                                                  
                                                  |                                                  
                       .                          |                                                  
              :=+*#*+*%*#=+*-                     |                ::=#=*+***++*+-                   
           -***--:  .   . --+#++:                 |            .-+=*+-. ...:   .=++*=:               
        :+%#-   . .:--=:=     .***-               |          .++*= .   : =-=-: .  .-=**-             
      :=*+.   .=**#*+#++****=   .=%*.             |        .=+=:....===*=*=+**=+== ...++=            
      *%=   :*#*-:    .    +*#-   -#=.            |       :+#-.  -+**    ... :::=##:  .-#=           
     +#-   =%*:   -=++*#=:  :*#:  .**-            |       +#:.. =*+.  .==+**+= :. *#   .*%-          
    :#=.  .%+.  :=#+. .-#+  .+%-   +%-    .       |      -#=   .%# . .-#-.:.=*-   =#-  .=#.    .     
    .-    +%=   -%+   +*.:.-*%-   :%*   :*%       |       .    :@: . -#-. .=+-..-+*-   -#=   .=*:    
          .@+   -#*   .+++***.  .-%+.   =#=       |            -@=.  :*+:: .--**+=:   :+*:  .-#=     
           +#=   -##- .       -=*#+:   +#+        |            .+#:.  :**:.        .-*#-. ..=%+.     
            +#+.   -+*+*==+***#*=.  .-+#=         |             .+#=....:**++*-+=-=+=.-.  =*%-       
            .-##*-      =--..    .:+##=           |              .:#+=:: :...:--    :-:==+%-         
               -+**=*-:: .. --==**#*-.            |                  +++++=-  .  .:===+*+-           
                  .:+++#+#+++=+=-.                |                     -=*=*++++*+*==               
                                                  |                           .                      
                                                  |                                                  
                                                  |                                                  

sliced W1   model 0.0258   gaussian baseline 0.1223
MMD^2       model 0.0007   gaussian baseline 0.0255
```

`--show-steps` prints the reverse process - pure noise on the left condensing into the spiral on the right:

```
        t=200          |         t=120          |          t=60          |          t=20          |          t=0          
       .....:..  .     |      .... .: : ..      |           .. .         |                        |                       
     .:.-=:-:::::. .   |     .:::--.-::-: .     |      :::-----:::       |         ::::.          |                       
    :----====---:-:..  |  . .::-==-=+=+=-.::.   | .  .--++*+#*++=-: .    |     .=+*##%#**=        |       +*#****+:       
 .::-=+++*#*++++---... |  .:.-==**+##**==--:. . |  .:-+*%#%@####**=-:    |   ..*%#***#*#+%#-      |     +%+-*+**+-%*      
  :-=-+*#%##%***+=:::. | .::-==*##%#%%#+*+=::   |  ::++##%%#@%%@%#*=::   |    +###%#*####*@+:     |    +*.##-*+:*#:@-     
.:::=+**%%@@%%#*+-=:.. | .::-==+##%#@##**+=-... |  .-=+#%%%%%%%%##*+=-:  |   :++#@#%#*##**%#*=.   |    +-+@.%=*+-#.@=-+   
.:::==**##%####+==::   |   :--=+**%%%%#*++=--.. |   .:=+###%@%@@%%*+=: . |    .:*###**#####*%+.   |      =%.*+=*+-**:%-   
 .:-====*###**+==-.::  |  :..-==+***#**+=--:::  |    :--=**#@%*%%*#-::.  |      :*%+*#%%####+:    |       +#=++*#*-*#-    
 ..::-==-++++=---:..   |  ..:::-=-===+=---.:... |    ..:====**#*+=+-.: . |       .=**%%%##+:      |        .***#**#=      
 ...:  :-=--.::.:..    |       .:.::-:-::.: .   |        .:-==-:: :      |         .::.:          |                       
        ....::  .   .  |       . :...: .        |           :  :.        |                        |
```

Sampling the saved model with DDIM in 25 steps instead of 200 (8x fewer network calls, a bit rougher):

```
3000 samples (DDIM, 25 steps) in 0.52s
                                                
                                                
                       .                        
               -+++==#*#+*+--                   
         . =****: -:    . . =*++.               
         -=+-::.-.:-::-:--...  -==.             
       +*=:::: ==*++:-=+%#+-. . .:#=            
      :+. .:.-*-:        : .-+--.:-@=.          
     **=-.  +=.  .::. =+*.   ++.  :=#:          
    -#=.---**-.. -#- . -*-::.+*..::-*  .:.      
     .-:. .#=: ::#+:  -*=:.:+++:- :%=. ::#+     
         .-#=    =#::. :***+=-. ..##: : :%-     
           +#     -*-. . .    ::+%---: :+-      
            =*=.  . .-====+=+=-*--.: -*@=       
             :**+.-.  -:-.-- .  ...+%*+         
               :-++-=-::--.: -*=+-=+..          
                   .-=-==#+=----                
                                                
                                                
                                                
sliced W1 vs 'spiral': 0.0433
```

Same settings on the other datasets (lower is better, the baseline is a gaussian with the data's mean and covariance):

| dataset | sliced W1 (model) | sliced W1 (gaussian) | MMD² (model) | MMD² (gaussian) |
|---|---|---|---|---|
| moons   | 0.0265 | 0.1547 | -0.0007 | 0.0611 |
| spiral  | 0.0258 | 0.1223 | 0.0007 | 0.0255 |
| rings   | 0.0329 | 0.0963 | 0.0010 | 0.0263 |
| gmm8    | 0.0180 | 0.2445 | 0.0001 | 0.1610 |
| checker | 0.0280 | 0.1260 | -0.0006 | 0.0293 |

(The MMD estimator is unbiased, so it can dip slightly below zero when the two sample sets are indistinguishable.)

## things I tripped over

- **The spiral melted into a blob** with a plain MLP on raw `(x, y)`. MLPs are biased toward smooth functions; adding a few octaves of `sin/cos` of the coordinates (the NeRF trick) fixed it immediately.
- **DDIM exploded** on the first step. With the cosine schedule `abar` at the final step is about 1e-5, so recovering `x0 = (x - sqrt(1-abar)·eps) / sqrt(abar)` multiplies any error in `eps` by ~300. Clipping the `x0` estimate (and re-deriving `eps` from it) is the standard fix.
- **EMA matters more than I expected.** Sampling from the raw weights at the end of training gives visibly noisier edges than the EMA copy.

## tests

```
pytest -q
```

The nicest test is the point-mass oracle: if the data is a single point `c`, the optimal noise predictor has a closed form, so plugging it in lets you check the sampler math exactly - both DDPM and DDIM have to collapse onto `c`. There's also a finite-difference gradient check on every parameter tensor of the network and a short end-to-end training run that has to beat the gaussian baseline.

## what I'd do next

- classifier-free guidance with a dataset label, so one model can draw any of the shapes on demand
- v-prediction and comparing it to eps-prediction at low step counts
- swap the MLP for a small transformer and try MNIST-sized images (that's where NumPy will start to hurt)

## license

MIT
