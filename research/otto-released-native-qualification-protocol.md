# Native qualification of the original released OTTO policy

21 September 2026. Qualify the original TensorFlow value model and unchanged published `RLPolicy` through an actor-owned public belief. This is an integration test, not an autonomous performance cohort or a new architecture experiment.

The corrected-runtime comparison completed all fixtures but the NumPy port disagreed on five selected actions at floating-point ties. Its action-equivalence requirement failed and remains unchanged. This qualification therefore uses the **original TensorFlow model and original policy on both sides**, with no NumPy-policy substitution or tie-rule relaxation.

## Public interface and runtime

The actor receives only a frozen public observation kernel and packets containing position, hit, done, step and in-bounds movement metadata. It constructs and owns its posterior. Never pass it the sampled-source environment, source position, seed, random stream, draw log or native posterior. The evaluator alone owns those objects and may inspect them for comparison.

All action IDs zero through three remain available. At a boundary, a blocked direction stays in place, consumes a step and produces another observation. The packet's `valid_actions` describes in-bounds movement; it is not a policy mask. Preserve the original observation update: zero the visited cell, multiply the relevant likelihood, clear the original tiny-negative interval, and divide only when total mass exceeds `1e-10`. Preserve the public found sentinel `-2`, terminal point-mass belief and reset lifecycle. Do not request another action or step after finding the source. Censoring is evaluator-owned and includes the last nonterminal update.

Use a new `.venv-otto-released-native` environment containing exactly the corrected reference distributions plus SciPy 1.18.1, which the native environment requires. Keep Python 3.12.13, NumPy 2.5.3, TensorFlow 2.20.0, tf-keras 2.20.1 and h5py 3.14.0. Keep prior runtimes unchanged. Record all installed distributions, CPU-only legacy-Keras float32 configuration and one numerical thread. Authenticate original source, actor, runner, tests, runtime, prior comparison closure and all inputs before native/model construction.

Load the previously authenticated original HDF5 copy as an input, without another output copy. Verify every loaded kernel and bias against the eight independently extracted tensors before any policy forward. This requires weight identity; it does not promote the failed NumPy policy to an equivalent implementation.

## Fixed mechanical fixtures

Both sensing regimes use N=53, two dimensions, four hit categories, Euclidean distance and R_dt=2. Sensing length is three for baseline and four for shift. Seeds **840001-840004** identify the four baseline fixtures and **840005-840008** the four shifted fixtures. They are fixed integration identifiers, not an untouched scientific evaluation set.

For each regime and initial hit one, two and three, set evaluator-only source position `(52,52)`. Prescribe the square actions `[0,2,1,3]` and observations `[0,1,2,3]`. Compare native and actor float64 beliefs and public metadata at reset and after every step. Censor after the fourth step only after incorporating its nonterminal observation.

For the fourth fixture in each regime, use initial hit one and evaluator-only source position `(26,27)`. Prescribe 26 north actions, a blocked north action, 26 west, a blocked west, 52 south, a blocked south, 52 east, a blocked east, then 26 north and 25 west to find the source. Prescribe zero hits on nonterminal steps; the native found branch supplies sentinel `-2`. Check all four stationary boundary transitions, exact step increments, observation handling, posterior updates and terminal behavior.

Source overrides and prescribed hits are intentionally mechanical. They test API/transition semantics and do not represent sampled gameplay or model effectiveness. Retain them explicitly in the evaluator record; they are never actor inputs. Newly constructed native kernels must match the already authenticated baseline/shift kernels byte-for-byte.

## Fixed policy comparisons and complete work

At four prefixes per regime, compare the original native-view policy against the actor-view policy: initial hit one at reset, initial hit two after the first square step, initial hit three after the fourth square step, and the boundary fixture after its first blocked north action. Reconstruct a separate public actor from the same saved public prefix for these checks, so a pending model choice does not contaminate subsequent prescribed actions.

Use symmetry averaging for every call. Record actual model inputs and branch masses from the original policy call. Require **byte-exact belief, centered inputs, masses and float32 costs, plus exact selected action**, under the unchanged first-action rule `abs(cost-min) < 1e-10`. No near-tie exemptions. Store all comparisons and input arrays.

The complete schedule has **eight native constructions/resets, 446 native steps and sixteen TensorFlow forwards**, comprising eight paired policy comparisons. It constructs sixteen public actors: eight for the mechanical paths and eight for the paired policy checks. The 446 path updates plus 64 public-prefix replay updates total 510 actor updates. Model calls from every actor are recorded; the mechanical-path actors must make none. Record attempted and returned calls before and after each operation in a durable ledger; record construction, graph build and weight loading separately. There are zero autonomous episodes, optimizer updates or NumPy-policy calls.

Freeze a source- and input-bound plan in Git before execution. Run once under the suspend-inclusive supervisor, limited to **600 seconds, 4 GiB RSS and 64 MiB output**. The pre-existing HDF5 and extracted tensors are authenticated inputs outside this output allowance. Preserve failures and partial work; do not rerun this plan to obtain a pass. A pass requires every fixed comparison, complete counts, unchanged source/input/runtime identities, successful worker and supervisor closure, absent process group, no late failure and independent saved-output readback.

Even a pass establishes only this fixed native/public integration. It does not establish autonomous competence, reproduce the historical TensorFlow runtime, admit a learned compact architecture or revise earlier failed scientific rules. A subsequent autonomous reference comparison needs its own frozen cohort, all-four-action contract and complete computation accounting.
