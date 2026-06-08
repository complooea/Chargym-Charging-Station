# CPU vs GPU Stepping in Mainstream DRL Environments

## Conclusion

Most mainstream deep reinforcement learning (DRL) environments primarily perform
environment stepping on the CPU. This is especially true for environments exposed
through the Gym/Gymnasium API, Stable-Baselines3-compatible environments,
classic MuJoCo/PyBullet robotics environments, Atari/ALE-style environments, and
many C++-accelerated benchmark environments such as Procgen.

The common architecture is:

1. The policy/value neural networks run on GPU when available.
2. The environment receives CPU-side actions, usually as NumPy arrays or Python
   scalars.
3. The environment computes `step()` dynamics on CPU.
4. The environment returns CPU-side observations/rewards/done flags.
5. The RL library batches those outputs and transfers tensors to GPU for model
   training or inference.

GPU-native stepping exists, but it is the exception rather than the default. It
typically requires an environment designed around batched tensor/JAX simulation,
such as NVIDIA Isaac Gym/Isaac Lab, Brax, MuJoCo MJX, or MuJoCo Warp.

## Evidence from Mainstream APIs

Gymnasium's environment API is intentionally general, but its examples and
space definitions are NumPy-centered. The documentation describes reset and step
observations as elements of the observation space and notes that these are
typically NumPy arrays. Its render API also defines RGB frame outputs as
`np.ndarray` objects. This strongly reflects a CPU/Python interoperability model
rather than a GPU tensor contract. See the Gymnasium `Env` documentation:
<https://gymnasium.farama.org/main/api/env/>.

Stable-Baselines3 keeps the same CPU-oriented boundary for vectorized
environments. Its VecEnv documentation says that `vec_env.step(actions)` expects
an array and returns `obs`, `rewards`, and `dones` as NumPy arrays with a batch
dimension. This means that even if the neural network runs on CUDA, the standard
SB3 environment interface is still NumPy-based at the environment boundary. See:
<https://stable-baselines3.readthedocs.io/en/md-doc/guide/vec_envs.html>.

Gymnasium's parallelization strategy also reinforces this model. The
`AsyncVectorEnv` documentation describes parallel environment execution using
Python `multiprocessing` processes and pipes, which is a CPU process-level
parallelism approach rather than GPU kernel-level stepping. See:
<https://gymnasium.farama.org/api/vector/async_vector_env/>.

## Evidence from Common Simulators and Benchmarks

MuJoCo's classic Python bindings expose the C engine through Python and expect
NumPy arrays or objects convertible to NumPy arrays for array inputs. Output
arrays written by MuJoCo must be writable NumPy arrays. The documentation also
shows the ordinary `mj_step(model, data)` loop as the base stepping mechanism.
See the MuJoCo Python documentation:
<https://mujoco.readthedocs.io/en/latest/python.html>.

MuJoCo's simulation documentation states that the classic simulation loop runs
single-threaded by default. This is still CPU execution, even though it is highly
optimized C/C++ execution rather than slow Python-only stepping. See:
<https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html>.

PyBullet similarly defaults to CPU physics. The PyBullet quickstart guide says
that by default PyBullet uses the Bullet 2.x API on CPU, while GPU/OpenCL Bullet
3.x support is described separately. See:
<https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html>.

Procgen is another useful mainstream example because it is optimized and
implemented mostly in C++, yet it is still CPU-based. Its README says the
environments run at thousands of steps per second on a single core, lists AVX as
a CPU requirement, and notes that the library does not require or use GPUs. See:
<https://github.com/openai/procgen>.

EnvPool improves environment throughput with a C++ batched environment pool,
pybind11, and a thread pool. Its documentation reports high Atari and MuJoCo
throughput using many CPU cores and explicitly compares itself with GPU-based
solutions like Brax and Isaac Gym. This positions EnvPool as a high-performance
CPU-side environment execution engine compatible with existing Gym-style RL
libraries. See:
<https://envpool.readthedocs.io/en/latest/>.

The Arcade Learning Environment (ALE), widely used for Atari DRL benchmarks,
exposes a Python interface that mirrors the C++ interface and steps games via
`ale.act(action)`. The documented interface is not a CUDA tensor interface. See:
<https://ale.farama.org/python-interface/>.

## Evidence from DRL Literature

The Brax paper makes the architectural split explicit: many existing simulation
engines run environment dynamics on CPU while RL algorithms run on GPU or TPU,
which introduces data-marshalling overhead between the simulator and learner.
The paper presents Brax as a response to that limitation by colocating physics
and optimization on accelerator hardware. See:
<https://openreview.net/pdf?id=VdvDlnnjzIN>.

This is a useful framing for the general DRL ecosystem: GPU-native stepping is
valuable enough that specialized projects advertise it as a major design
feature. If GPU stepping were already the mainstream default, projects such as
Brax and Isaac Gym would not need to distinguish themselves around end-to-end
accelerator execution.

## GPU-Native Exceptions

NVIDIA Isaac Gym is a major exception. NVIDIA describes it as an end-to-end GPU
RL pipeline where physics simulation, observation computation, reward
computation, and PyTorch policy training can all remain on GPU. Its tensor API
is designed to avoid CPU-GPU transfer bottlenecks and to support many
simultaneous environments on a single GPU. See:
<https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/>.

Brax is another exception. Its README describes Brax as a JAX-based physics
engine designed for accelerator hardware, including GPU and TPU execution, with
massively parallel simulation. See:
<https://github.com/google/brax>.

Modern MuJoCo also now documents GPU-accelerated backends: MJX, based on JAX,
and MuJoCo Warp, based on NVIDIA Warp. These backends are distinct from the
classic C-engine Python stepping path and are aimed at large-scale parallel
simulation on accelerators. See:
<https://mujoco.readthedocs.io/en/stable/overview.html>.

## Implication for `ChargingEnv`

`Chargym_Charging_Station/envs/Charging_Station_Enviroment.py` follows the
mainstream Gym/SB3 pattern. Its `step()` method calls helper code built from
NumPy arrays, Python loops, Python `min`/`max`/`sum`, and SciPy `.mat` file I/O
at episode end. Its observation and action spaces are Gym `Box` spaces with
NumPy `float32` arrays.

Therefore, the environment currently steps on CPU. Training with a GPU can still
accelerate the neural network part of DDPG/PPO, but it does not make
`env.step()` GPU-native. Passing a CUDA tensor into this environment would not
change the stepping backend; it would either fail at the Gym/SB3 boundary or
force scalar/device-to-host conversions, which is usually slower.

For this environment, GPU-native stepping would require a deliberate redesign:

1. Store environment state as batched `torch.Tensor`, JAX arrays, or CuPy arrays.
2. Replace per-car Python loops with vectorized tensor operations.
3. Avoid NumPy/SciPy I/O in the hot stepping path.
4. Batch many independent charging-station environments together so the GPU has
   enough parallel work.
5. Use an RL stack that can consume GPU-resident observations/actions without
   converting through NumPy.

Given the current problem size of 10 cars and 24 hourly timesteps per episode,
CPU stepping is not unusual and is likely the pragmatic default. The better
optimization path is usually CPU vectorization, multiprocessing/vectorized
environments, or batched rollouts before attempting a full GPU-native rewrite.

## Source Index

- Gymnasium `Env` API: <https://gymnasium.farama.org/main/api/env/>
- Gymnasium `AsyncVectorEnv`: <https://gymnasium.farama.org/api/vector/async_vector_env/>
- Stable-Baselines3 VecEnv API: <https://stable-baselines3.readthedocs.io/en/md-doc/guide/vec_envs.html>
- MuJoCo Python bindings: <https://mujoco.readthedocs.io/en/latest/python.html>
- MuJoCo simulation loop: <https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html>
- MuJoCo overview and GPU backends: <https://mujoco.readthedocs.io/en/stable/overview.html>
- PyBullet quickstart guide: <https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html>
- Procgen README: <https://github.com/openai/procgen>
- EnvPool documentation: <https://envpool.readthedocs.io/en/latest/>
- ALE Python interface: <https://ale.farama.org/python-interface/>
- Brax paper: <https://openreview.net/pdf?id=VdvDlnnjzIN>
- Brax README: <https://github.com/google/brax>
- NVIDIA Isaac Gym technical blog: <https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/>
