# Independent commitment-audit preflight

The final auditor and test source pins below passed 16 synthetic tests and Ruff. No real prediction arrays, model calls, encoder calls or training calls were used.

These files preserve text already returned by the command tool, copied into this collection after the tests finished. They are not newly executed tests or independently timestamped raw process logs. The initial 14-test pass and single Ruff BLE001 finding are retained. The initial source snapshot was not separately pinned. Before the final run, the checker was narrowed to selected metric payload authentication, an explicit annotation documented the intentional failure-preservation exception handler, and two synthetic integration/authentication tests were added. The final 16-test pass and clean Ruff output apply to the final source hashes recorded in receipt.json.

The tests cover closed-form probability swapping, concentration-dependent hard decisions, two-candidate identity, extreme finite log probabilities, exact ties, selected-candidate branches, grouping and rare supports, repair/harm partitions, corrupt inputs/manifests, exclusive failure preservation, and an artificial nine-input producer-to-independent-checker integration. The real diagnostic audit remains unexecuted at this preflight stage.
