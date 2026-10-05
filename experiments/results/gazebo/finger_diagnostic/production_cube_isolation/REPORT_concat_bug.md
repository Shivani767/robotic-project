# Production cube isolation: does adding the cube reproduce the stall?

Config: production world via `world_sdf(seed=3)` with `robot_urdf` embedded, real
table, cube asserted present at the episode pose [0.02259, 0.22558, 0.008], arm
commanded to the episode step-56 grasp joints, then ONLY both fingers swept
0..40 mm per joint. No policy, no state machine, no mid-sweep teleport.

## Result

full_stroke: True  |  max_q_m: 0.04000000000035143  |  stall_row: {'commanded_q_m': 0.01, 'measured_q_m': 0.0056802294813523375}
joint_state: True  finger_ft: False (absent: production model has no FT sensor)
object_contacts topic: True  table_contacts topic: True

cmd     left_q    right_q   LvsCube RvsCube FvsTab CvsTab
 0.000  0.0006255558578534422 0.0004195466643129568 False   True    True   True  
 0.005  0.0023070096009952874 0.005329834982679679 False   True    True   True  
 0.010  0.0056802294813523375 0.010587264204971553 False   True    True   True  
 0.012  0.008711403995556559 0.012657047987523866 False   True    True   True  
 0.014  0.011324967886543828 0.014519707333625044 False   True    True   True  
 0.016  0.013905220366664429 0.015992622291327294 False   True    True   True  
 0.018  0.01658111784950263 0.01815177261838482 False   False   False  True  
 0.020  0.018608383897886945 0.020430573827929552 False   True    True   True  
 0.021  0.019610129027882967 0.021419454090609382 False   True    True   True  
 0.022  0.020608785487396097 0.022664926776022134 False   True    True   True  
 0.024  0.02241252429213467 0.024577096300184276 False   True    True   True  
 0.026  0.024606473474274443 0.026171464038986396 False   True    True   True  
 0.030  0.0286184970874887 0.027152146373372245 False   True    True   True  
 0.035  0.03374785611613009 0.03251073420639245 False   True    True   True  
 0.040  0.04000000000035143 0.03795017706002826 False   False   False  True  

Arm settled within ~2 mrad of the episode joints on joints 1-4 and 6, 13 mrad on
joint5, 26 mrad on joint7. Cube stayed on the table throughout (cube_vs_table=True).
Right finger was in reported contact with the cube at nearly every sweep point, yet
neither finger stalled; both reached the full stroke.

## Verdict: CASE 2
The cube is present at the failing pose and the fingers track. The cube is NOT
sufficient to reproduce the stall. The remaining difference is the exact episode
state: the policy-held arm pose, the tick-by-tick command sequence, or the
episode's own contact instrumentation, which reported zero finger<->cube contacts
in a scene where this run reports right-finger contact at nearly every point.

Gap recorded, not glossed: pad world poses were not captured (no pose topics in
this world), so brief section 4's minimum-separation calculation is not done, and
the finger FT sensor is absent from the production model so the ~5 N reaction has
no per-row measurement here.

Root cause: NOT IDENTIFIED. Part 1 remains open.