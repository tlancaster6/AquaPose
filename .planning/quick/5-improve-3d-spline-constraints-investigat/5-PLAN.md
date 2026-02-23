---
phase: quick-5
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md
autonomous: true
requirements: [QUICK-5]

must_haves:
  truths:
    - "Report identifies why current regularization allows near-180-degree folds"
    - "Report quantifies the effective penalty magnitude at various fold angles"
    - "Report recommends concrete parameter or structural changes to prevent folding"
  artifacts:
    - path: ".planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md"
      provides: "Investigation report on spline folding and regularization"
      min_lines: 50
  key_links: []
---

<objective>
Investigate why the CurveOptimizer's regularization allows splines to fold nearly in half, and report findings with recommendations.

Purpose: The optimizer's curvature and smoothness penalties should prevent non-physical shapes, but in practice splines can bend ~180 degrees. This investigation identifies the root cause and recommends fixes.
Output: A diagnostic report with analysis, numerical evidence, and actionable recommendations.
</objective>

<execution_context>
@C:/Users/tucke/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/tucke/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/aquapose/reconstruction/curve_optimizer.py
@tests/unit/test_curve_optimizer.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Analyze regularization mechanics and quantify folding vulnerability</name>
  <files>.planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md</files>
  <action>
Write a Python diagnostic script (inline, not saved) that:
1. Constructs control points for various fold severities (0, 30, 60, 90, 120, 150, 180 degree bends) using 4-point and 7-point B-spline configurations.
2. For each configuration, computes:
   - `_curvature_penalty` value
   - `_smoothness_penalty` value
   - `_length_penalty` value (using default config)
   - Total regularization loss (weighted sum with default lambdas)
   - The ratio of regularization loss to a typical data loss (use 5-15px chamfer as reference)
3. Analyzes the B-spline basis smoothing effect: evaluate the actual curve from folded control points and measure the maximum bend angle on the EVALUATED curve (not just control points). This reveals whether basis smoothing masks the fold from the penalty.
4. Checks whether the curvature penalty's quadratic-beyond-threshold formulation creates sufficient gradient at extreme angles vs the data loss gradient.

Write a comprehensive report (5-REPORT.md) covering:
- **Current Architecture**: How each penalty works (curvature, smoothness, length), their formulations, default weights
- **Root Cause Analysis**: Why folding happens despite penalties. Key areas to investigate:
  a. The curvature penalty operates on CONTROL POINTS, not the evaluated curve -- B-spline smoothing can redistribute a sharp fold across multiple segments, each individually under the 30-degree threshold
  b. With K=4 coarse control points, there are only 2 interior angles (K-2=2). A 180-degree fold can be distributed as two 90-degree bends, each penalized but the quadratic penalty at 60 degrees excess (90-30) may be too weak relative to data loss
  c. The smoothness penalty (second differences) penalizes acceleration but not total curvature -- a smooth U-turn has low second-difference
  d. The length penalty has a wide tolerance band (+-30%) and only activates outside it
  e. The lambda weights (curvature=5.0, smoothness=1.0) may be too low relative to typical chamfer losses
- **Numerical Evidence**: Table of penalty values at each fold angle for both K=4 and K=7
- **Recommendations**: Concrete changes ranked by impact, such as:
  a. Add a total turning angle constraint (sum of all bend angles < threshold)
  b. Evaluate curvature on the dense evaluated curve, not just control points
  c. Increase lambda_curvature significantly (e.g., 50-100x)
  d. Add a chord-to-arc-length ratio penalty (straight fish: ratio ~1.0, folded: ratio << 1.0)
  e. Lower max_bend_angle_deg for adjacent segments
  f. Add a monotonicity constraint on the tangent direction (prevents reversal)
  </action>
  <verify>
The report file exists at `.planning/quick/5-improve-3d-spline-constraints-investigat/5-REPORT.md`, contains numerical evidence tables, and lists at least 3 ranked recommendations with implementation difficulty estimates.
  </verify>
  <done>
Report delivered with: (1) clear explanation of why folding occurs, (2) numerical penalty values at various fold angles showing the weakness, (3) at least 3 concrete recommendations ranked by impact and implementation effort.
  </done>
</task>

</tasks>

<verification>
- Report exists and is comprehensive (50+ lines)
- Numerical evidence is included (not just qualitative analysis)
- Recommendations are actionable (specific code changes, not vague suggestions)
</verification>

<success_criteria>
- Root cause of spline folding is identified with numerical evidence
- At least 3 recommendations with impact/effort ranking
- Report is self-contained and actionable for a follow-up implementation task
</success_criteria>

<output>
After completion, create `.planning/quick/5-improve-3d-spline-constraints-investigat/5-SUMMARY.md`
</output>
