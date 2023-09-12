# Minimal implementation of MACAW (ICML 2021)
To run the code in the Google colab follow these steps:

 

 1. Clone the repository and install dependencies:

	    !git clone https://github.com/Arya-Ebrahimi/macaw-min.git
	    %cd macaw-min
	    !pip install -r requirements.txt
   
2. Download Mujoco210:

	   %cd macaw-min
	   !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dGltHS74krx4cC-3gdv2ReMrEuFsfeR1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dGltHS74krx4cC-3gdv2ReMrEuFsfeR1" -O mujoco && rm -rf /tmp/cookies.txt
	   !unzip mujoco
	   !mkdir /root/.mujoco && cp -r mujoco210 /root/.mujoco
   
  3. Download the apt dependencies using the following commands:
   

		 !apt-get install libosmesa6-dev
		 !apt-get install patchelf

  4. Download the offline data:
	  

		 !wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HWelNwEfKRqmduqVXLA42vNkwDKBWV9w' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HWelNwEfKRqmduqVXLA42vNkwDKBWV9w" -O data && rm -rf /tmp/cookies.txt
		 !unzip data
3. Run the code:

	   !export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin && python impl.py


This code trains MACAW on the simple Cheetah-Direction problem, which has only two tasks (forwards and backwards). `impl.py` contains example of loading the offline data (`build_networks_and_buffers`) and performing meta-training (loop in `run.py`). `losses.py` contains the MACAW loss functions for adaptation the value function and policy. `utils.py` contains the replay buffer implementation that loads the offline data.
