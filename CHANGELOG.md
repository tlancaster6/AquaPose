# CHANGELOG

<!-- version list -->

## v1.1.0-dev.1 (2026-02-25)

### Bug Fixes

- Add NaN check for observed points in residual computation
  ([`cd9b78b`](https://github.com/tlancaster6/AquaPose/commit/cd9b78b192147c61f82bea342af9623500cc0fa5))

- Add warm-start consistency check and orientation-invariant GT metric
  ([`c818fba`](https://github.com/tlancaster6/AquaPose/commit/c818fba60a9b413e5c66cca7fcaea51a1ec3e03e))

- Boundary clip check and birth index mapping bugs
  ([`4cbad9a`](https://github.com/tlancaster6/AquaPose/commit/4cbad9af00f5f04880b9bcc3adac1dd6ae2d9898))

- Collect optimizer snapshots for first valid frame, not just frame 0
  ([`2779458`](https://github.com/tlancaster6/AquaPose/commit/27794589d33024bdb3491ac7f271c37eef2bb453))

- Commiting pre-refactor codebase state
  ([`71f0ef4`](https://github.com/tlancaster6/AquaPose/commit/71f0ef4f08e0a576922dbe7f48ad7027e3f1c7be))

- Decouple snap_threshold from inlier_threshold in triangulation
  ([`c1cd279`](https://github.com/tlancaster6/AquaPose/commit/c1cd2791218cec216120d27132052b28dd30c4e8))

- Propagate inlier_threshold to _pairwise_chord_length
  ([`8745f6b`](https://github.com/tlancaster6/AquaPose/commit/8745f6b47f85b42d6b7d20499c2ffa937bcbb973))

- Re-triangulate RANSAC centroid seeds using all inlier cameras
  ([`083ec7c`](https://github.com/tlancaster6/AquaPose/commit/083ec7cc19d54819f4adb416db0661c41f43010e))

- Replace greedy orientation alignment with brute-force enumeration
  ([`7450d50`](https://github.com/tlancaster6/AquaPose/commit/7450d50997ffc7ceaca7c39ebdd94a74a2501fb4))

- Spline-based residuals, per-camera breakdown, and 3D plot scaling
  ([`e6b494d`](https://github.com/tlancaster6/AquaPose/commit/e6b494d6c840bd20c169cd2f564029d44c687378))

- Use multi-camera reprojection residual for orientation scoring
  ([`489d7a1`](https://github.com/tlancaster6/AquaPose/commit/489d7a166d681c2cdf704d6cbfbe90279f8dcd04))

- **01**: Revise plans based on checker feedback
  ([`19266ca`](https://github.com/tlancaster6/AquaPose/commit/19266caa05cc2f78f6739fd01beb1be519868ad5))

- **02**: Revise plans based on checker feedback
  ([`697e36f`](https://github.com/tlancaster6/AquaPose/commit/697e36f8417e4700732a32fc4558ecbec784fba4))

- **02**: Revise plans based on checker feedback
  ([`c8895e6`](https://github.com/tlancaster6/AquaPose/commit/c8895e65972e0a570467fe9994624a214bf119ce))

- **02**: Revise plans based on checker feedback
  ([`5ce5818`](https://github.com/tlancaster6/AquaPose/commit/5ce58182d9e28b97bebb0671511e8067f610dd88))

- **09-02**: Make upsampled control points contiguous for L-BFGS
  ([`8631c94`](https://github.com/tlancaster6/AquaPose/commit/8631c944a915523ecc56b3993063afa546e02582))

- **curve-optimizer**: Fix loss scaling, remove Huber wrapping, use absolute convergence
  ([`38d43dc`](https://github.com/tlancaster6/AquaPose/commit/38d43dca7c0bfb50f526291481b1c7e24596357a))

- **curve-optimizer**: Reject bad triangulation seeds and add depth penalty fallback
  ([`0598904`](https://github.com/tlancaster6/AquaPose/commit/059890440e39cf9275f5faa63c9f2ab19fc7efb1))

- **tracking**: Eliminate jerk-freeze-die cycle via depth validity and anchored single-view claims
  ([`b1e4ad4`](https://github.com/tlancaster6/AquaPose/commit/b1e4ad4c18988688161cb80cc7433929cccc175a))

- **tracking**: Fix coasting prediction stall causing jerk-freeze-die cycle
  ([`cde210b`](https://github.com/tlancaster6/AquaPose/commit/cde210ba285202b5119e1e6149dc6b8d45fcdb7a))

- **tracking**: Resolve phantom track births in real-calibration diagnostic
  ([`cc70ae1`](https://github.com/tlancaster6/AquaPose/commit/cc70ae114894fc8ce4d352c385e0a40313261716))

- **triangulation**: Add three-layer defence against above-water outliers
  ([`c2e1bba`](https://github.com/tlancaster6/AquaPose/commit/c2e1bba39949aabaaeb0975e7cdaf32265a0a44c))

### Chores

- Planning doc updates and organizing
  ([`d00f49c`](https://github.com/tlancaster6/AquaPose/commit/d00f49c5638f535b66a3fd95eefa94278d143049))

- Reorienting planning to focus on a less computationally expensive alternative reconstruction route
  ([`5d7c40e`](https://github.com/tlancaster6/AquaPose/commit/5d7c40e96cdd7e2db332a01bbda7e0e3f25dc48a))

- **02-01**: Delete debug and exploration scripts from scripts/
  ([`d366610`](https://github.com/tlancaster6/AquaPose/commit/d36661044836da162e6476a208786ff972246407))

- **04.1-01**: Archive Phase 4 ABS code and strip optimization module
  ([`81b52af`](https://github.com/tlancaster6/AquaPose/commit/81b52af9545f9d7a76543922b6e7458aeadc7818))

- **04.1-01**: Update CLAUDE.md to reflect triangulation pivot and removed optimization module
  ([`e352ad5`](https://github.com/tlancaster6/AquaPose/commit/e352ad50f1b5ffc1ff78785d0db0eb5a39b995c2))

- **04.1-01**: Update pyproject.toml description to reflect triangulation pivot
  ([`c606ab7`](https://github.com/tlancaster6/AquaPose/commit/c606ab70098cf7fa83d214eb1e15345a8ea70546))

- **deps**: Bump actions/checkout from 4 to 6
  ([`4b92bf3`](https://github.com/tlancaster6/AquaPose/commit/4b92bf30ee1ab54777b870cf584e4a21709e686c))

- **deps**: Bump actions/download-artifact from 4 to 7
  ([`46aeb13`](https://github.com/tlancaster6/AquaPose/commit/46aeb1308f5162f555d21ce4fa5aa8f6ee1b5f42))

- **deps**: Bump actions/setup-python from 5 to 6
  ([`4c93562`](https://github.com/tlancaster6/AquaPose/commit/4c9356268baac0fc830955a3a771f08772625f90))

- **deps**: Bump actions/upload-artifact from 4 to 6
  ([`c0943c4`](https://github.com/tlancaster6/AquaPose/commit/c0943c47cc729c17d5e7f2247d2540ef32642bdd))

### Code Style

- **04-01**: Rename uppercase locals in loss.py for linter compliance
  ([`f565d38`](https://github.com/tlancaster6/AquaPose/commit/f565d38587fde93532fc08b765cfd73666a1541d))

### Documentation

- Archive ghost-track-phantom-births debug session as resolved
  ([`2bd4a76`](https://github.com/tlancaster6/AquaPose/commit/2bd4a760e5f38ee598bbc457ba6de77a2df29038))

- Capture todo - Active calibration refinement
  ([`3bbaa99`](https://github.com/tlancaster6/AquaPose/commit/3bbaa9974745bf6cd32dea95b5092d43f9034e08))

- Capture todo - Adaptive depth range for epipolar refinement
  ([`0f2988c`](https://github.com/tlancaster6/AquaPose/commit/0f2988c14c13b2b422d83e08400645ac33051095))

- Capture todo - Add YOLO-OBB support
  ([`2e82274`](https://github.com/tlancaster6/AquaPose/commit/2e82274b9a9b69d4bbc7e53b6d548f0ec5345a46))

- Capture todo - Consolidate scripts into CLI workflow
  ([`c978d22`](https://github.com/tlancaster6/AquaPose/commit/c978d224b46fd8f9c43d5c7537243a3ad9a6ae9f))

- Capture todo - Integrate full-frame exclusion masks from AquaMVS
  ([`a16e211`](https://github.com/tlancaster6/AquaPose/commit/a16e21123033f53ebfdaacf2ae3aa18e770f1004))

- Capture todo - Keypoint-based pose estimation alternative to segmentation
  ([`c870c2e`](https://github.com/tlancaster6/AquaPose/commit/c870c2e6dc8749631bd5786443116fea638f310e))

- Capture todo - Windowed velocity smoothing for tracking priors
  ([`f920fa6`](https://github.com/tlancaster6/AquaPose/commit/f920fa67c3273bbdaaf3f92d5b52db96e2ff348c))

- Resolve debug session curve-optimizer-high-loss
  ([`2682b03`](https://github.com/tlancaster6/AquaPose/commit/2682b03864d1b10a63cf853bf25c1fe48dd9e380))

- Resolve debug session curve-optimizer-zero-loss
  ([`f8d1c97`](https://github.com/tlancaster6/AquaPose/commit/f8d1c970ec8984a3aa409947677f4735edc90748))

- Resolve debug session triangulation-above-water-outliers
  ([`bd20d03`](https://github.com/tlancaster6/AquaPose/commit/bd20d03eca906c8abf6d918c3e5308776f302225))

- Resolve debug track-jerk-freeze-die
  ([`8be7672`](https://github.com/tlancaster6/AquaPose/commit/8be76729ff214682fc168d0f51b50ce0d4f0f933))

- Resolve debug track-jerk-freeze-die-v2
  ([`3637db8`](https://github.com/tlancaster6/AquaPose/commit/3637db85366c756ae944ddafe469cd4d77b131ac))

- Revert SEG-04 to Pending — full crop pipeline not yet verified
  ([`77469fe`](https://github.com/tlancaster6/AquaPose/commit/77469fef50784f9c4ac906dd4de0f37024e08233))

- **01**: Capture phase context
  ([`4718ce2`](https://github.com/tlancaster6/AquaPose/commit/4718ce29d2316850a97a02aefb72d1fcf7857366))

- **01**: Create phase plan
  ([`51559b4`](https://github.com/tlancaster6/AquaPose/commit/51559b4ae423537c721a208f0909a40dfec91283))

- **01**: Research phase calibration and refractive geometry
  ([`d3e52ae`](https://github.com/tlancaster6/AquaPose/commit/d3e52ae403c94fa4822528d25a484add41f160fa))

- **01-01**: Complete calibration loader and refractive projection plan
  ([`f2c164b`](https://github.com/tlancaster6/AquaPose/commit/f2c164b1951fbc85788bad29584ba105af80519a))

- **01-02**: Complete Z-uncertainty characterization plan
  ([`0337394`](https://github.com/tlancaster6/AquaPose/commit/033739459aa616d299674fb9d9c1a881df4e83b8))

- **02**: Add directive to scan scripts/ for reusable code
  ([`14f5f8e`](https://github.com/tlancaster6/AquaPose/commit/14f5f8eab11b844104425cc08070ea0dfd02aa52))

- **02**: Capture phase context
  ([`c702e1a`](https://github.com/tlancaster6/AquaPose/commit/c702e1a4d01901b0789887750a2e581d310e9242))

- **02**: Create phase 2 segmentation pipeline plans
  ([`2472dfe`](https://github.com/tlancaster6/AquaPose/commit/2472dfe2af2ebd0066055b1ce01ca908eb0c7585))

- **02**: Create phase plan — cleanup, pseudo-labels, Mask R-CNN, integration
  ([`4304c95`](https://github.com/tlancaster6/AquaPose/commit/4304c95c8b9100bf3ffadfcb629f48ca66beae85))

- **02**: Remove female-specific accuracy references (no sex labels available)
  ([`a2b457b`](https://github.com/tlancaster6/AquaPose/commit/a2b457b7120fb814030eb7f7c47b8d6a1f8a0d3f))

- **02**: Remove outdated continue-here handoff file
  ([`cf1cc2e`](https://github.com/tlancaster6/AquaPose/commit/cf1cc2e2ba741aea37a507c6be07e40dc8a158c3))

- **02**: Research segmentation pipeline domain
  ([`d5f65c6`](https://github.com/tlancaster6/AquaPose/commit/d5f65c68120c791505f60c95881e09c1631e159b))

- **02**: Update phase context with revised pipeline decisions
  ([`42aa012`](https://github.com/tlancaster6/AquaPose/commit/42aa01270ff8d5f61c55eb9838c6347f6a17b0c1))

- **02-01**: Complete cleanup plan — Label Studio removed, debug scripts deleted
  ([`e6bbc5b`](https://github.com/tlancaster6/AquaPose/commit/e6bbc5b09eb5b47097663b1e45c56336ab93a508))

- **02-01**: Complete plan summary and update state
  ([`7c3940f`](https://github.com/tlancaster6/AquaPose/commit/7c3940fa3d9cb937199be038b4f2f65168d3e2cd))

- **02-02**: Complete plan summary
  ([`7bd7188`](https://github.com/tlancaster6/AquaPose/commit/7bd71887b4756eeab7d8814fa0e8634a4956ce28))

- **02-02**: Complete pseudo-labeling pipeline update plan
  ([`d557267`](https://github.com/tlancaster6/AquaPose/commit/d55726742e1f3b8e158bdcd57cbcf315574c4a93))

- **02-03**: Complete Mask R-CNN inference refactor and training update plan
  ([`b705105`](https://github.com/tlancaster6/AquaPose/commit/b7051056a0b86082e306abb749eb90b8bdfef945))

- **02-03**: Complete plan summary and update state
  ([`2fd238d`](https://github.com/tlancaster6/AquaPose/commit/2fd238d87c8b1b125b3608e16acdb43941f45657))

- **02.1**: Capture phase context
  ([`1f7035d`](https://github.com/tlancaster6/AquaPose/commit/1f7035d702432557fc58c343e0cd621158a7a280))

- **02.1**: Create phase plan
  ([`d6ab572`](https://github.com/tlancaster6/AquaPose/commit/d6ab572eba0b44f5c378c219dec9d3d8c42fd2f5))

- **02.1**: Research phase domain
  ([`ccd2694`](https://github.com/tlancaster6/AquaPose/commit/ccd26944bcb46487768d9349b13cc68ce9cf7a6d))

- **02.1**: Revise plan 02 to use YOLO detection instead of MOG2
  ([`7febd3a`](https://github.com/tlancaster6/AquaPose/commit/7febd3a89da02f1c7974e6803c496a202576f8ac))

- **02.1-01**: Complete MOG2 diagnostic plan - annotated stills produced for 2 cameras
  ([`2214888`](https://github.com/tlancaster6/AquaPose/commit/2214888ce1a0986cf46427e932f555787442d4bc))

- **02.1-02**: Complete plan 02 — SAM2 evaluation script, awaiting human-verify checkpoint
  ([`92b3e36`](https://github.com/tlancaster6/AquaPose/commit/92b3e368810ce7c89ed4ad809392bd62134bd967))

- **02.1-02**: Update state to checkpoint at Task 2 human-verify
  ([`d651472`](https://github.com/tlancaster6/AquaPose/commit/d651472bd89bba0d51b39e440412960d33997de5))

- **02.1-03**: Skip Mask R-CNN plan — superseded by U-Net pipeline
  ([`a2cdf52`](https://github.com/tlancaster6/AquaPose/commit/a2cdf5283bb75815fc16ba33a72917c8cbf0b31b))

- **02.1.1**: Capture phase context
  ([`9a768f7`](https://github.com/tlancaster6/AquaPose/commit/9a768f7d79dc170924a33f8372ba6be4ead0205d))

- **02.1.1**: Create phase plan
  ([`ca6f9e4`](https://github.com/tlancaster6/AquaPose/commit/ca6f9e44fcbb34efff68725e844137baf854158c))

- **02.1.1**: Create plan 03 — pipeline integration
  ([`8cc870d`](https://github.com/tlancaster6/AquaPose/commit/8cc870da712dcc40a33cf5a58958c65def120883))

- **02.1.1**: Research phase domain - YOLO training and pipeline integration
  ([`8236853`](https://github.com/tlancaster6/AquaPose/commit/8236853f5fe2418bda56bdc8643efc912980cfc0))

- **02.1.1-01**: Complete plan 01 — frame sampling script built, at checkpoint:human-verify
  ([`071c326`](https://github.com/tlancaster6/AquaPose/commit/071c32658786c4f7c6f3571d1e3faab1c5366846))

- **02.1.1-01**: Complete plan 01 — YOLO dataset ready (120 train / 30 val)
  ([`0e16961`](https://github.com/tlancaster6/AquaPose/commit/0e1696174c6714a4664b8ae61aeeedac99d468a5))

- **02.1.1-02**: Complete plan 02 — YOLO verified, pipeline integration ready
  ([`41851b1`](https://github.com/tlancaster6/AquaPose/commit/41851b16578ab34de0471784ab9e98afda6786a4))

- **02.1.1-02**: Complete plan 02 — YOLODetector and eval scripts at checkpoint
  ([`05ace03`](https://github.com/tlancaster6/AquaPose/commit/05ace03144262d4d50b02599d2b8cf64d2c88d8a))

- **02.1.1-03**: Complete plan 03 — YOLO wired into SAM2 pipeline integration
  ([`2865bf4`](https://github.com/tlancaster6/AquaPose/commit/2865bf49066f05c29d62c090a33be68043c1b99e))

- **03**: Capture phase context
  ([`db0b22c`](https://github.com/tlancaster6/AquaPose/commit/db0b22cfd50eb68e17c6d7cbab6da47e8cb5e277))

- **03**: Create phase plan
  ([`eb598a2`](https://github.com/tlancaster6/AquaPose/commit/eb598a2f45719c2b5bb7c1bea86890c66e10ae4b))

- **03**: Research phase domain
  ([`30ff2aa`](https://github.com/tlancaster6/AquaPose/commit/30ff2aad622caae91765702fa839d8f9e5f6c27d))

- **03-01**: Complete parametric fish mesh plan summary and update state
  ([`cb73b45`](https://github.com/tlancaster6/AquaPose/commit/cb73b458b55ab029a2de1cfa0656b5ef07a28401))

- **03-02**: Complete initialization pipeline plan summary and update state
  ([`7d7e62b`](https://github.com/tlancaster6/AquaPose/commit/7d7e62bb6b4cac1eda35e2ff4fcc3317f44b7f8f))

- **04**: Capture phase context for single-fish reconstruction
  ([`44ca75c`](https://github.com/tlancaster6/AquaPose/commit/44ca75c2e6d9fdfb5c0e85d6c679c9dd725dbc83))

- **04**: Create phase plan for per-fish reconstruction
  ([`ba7031e`](https://github.com/tlancaster6/AquaPose/commit/ba7031efa60d8fa87b3f0f1c0c7eb8822154d4dc))

- **04**: Rename Single-Fish to Per-Fish Reconstruction
  ([`b50a9f8`](https://github.com/tlancaster6/AquaPose/commit/b50a9f85e95250d733fe826d8652e0f02ec838be))

- **04**: Research phase domain
  ([`a005676`](https://github.com/tlancaster6/AquaPose/commit/a00567618f80ce515778819621935d8153521430))

- **04-01**: Complete renderer and loss plan — SUMMARY, STATE, ROADMAP, REQUIREMENTS
  ([`7775841`](https://github.com/tlancaster6/AquaPose/commit/7775841f0cf99eb822604d0cf4e901102072cc0b))

- **04-02**: Complete FishOptimizer plan — SUMMARY, STATE, ROADMAP, REQUIREMENTS
  ([`99fe493`](https://github.com/tlancaster6/AquaPose/commit/99fe49387dc86fa92885868bc9d1ca87cfaf3f50))

- **04-03**: Complete holdout validation plan — awaiting human verification
  ([`db326d0`](https://github.com/tlancaster6/AquaPose/commit/db326d0f03002e62880c3d47ebcbe26de62ccd84))

- **04.1**: Capture phase context
  ([`1465b1f`](https://github.com/tlancaster6/AquaPose/commit/1465b1f6935fce586ba13a22c783a9a0b9b32e53))

- **04.1**: Create phase plan
  ([`fae8ce8`](https://github.com/tlancaster6/AquaPose/commit/fae8ce83b6ecf19def6ae29813ad2adaab993bad))

- **04.1**: Research phase domain
  ([`9d9524d`](https://github.com/tlancaster6/AquaPose/commit/9d9524dda9b9ee9b37f8de9231050b7bb95a8d7c))

- **04.1-01**: Complete plan - archive Phase 4 ABS, strip optimization module
  ([`f504cc7`](https://github.com/tlancaster6/AquaPose/commit/f504cc72fe9b13fcdffa648e3f178f2418714325))

- **05**: Capture phase context
  ([`6a98e58`](https://github.com/tlancaster6/AquaPose/commit/6a98e5825c68f6984ce2ccff3acbba4bff2d5d2b))

- **05**: Create phase plan
  ([`dc4cce5`](https://github.com/tlancaster6/AquaPose/commit/dc4cce517ddf776e874bfb5f0ded9c966255a0b3))

- **05**: Research phase domain
  ([`e1d828b`](https://github.com/tlancaster6/AquaPose/commit/e1d828b2f73199b72c0567785d9e3dbd7480d75b))

- **05-01**: Complete plan - RANSAC centroid ray clustering
  ([`e4a8bf9`](https://github.com/tlancaster6/AquaPose/commit/e4a8bf987d4395a9be66719c50abd7dbc606f4ea))

- **05-02**: Complete plan - Hungarian tracker with SORT lifecycle
  ([`d81dbc5`](https://github.com/tlancaster6/AquaPose/commit/d81dbc53c8c218ce3ec36d969560bae1e1af95e7))

- **05-03**: Complete HDF5 writer plan - tracking module finalized
  ([`b205609`](https://github.com/tlancaster6/AquaPose/commit/b205609bdcee23ac40e1a190333f5955b8fe979c))

- **06**: Add boundary-clipped masks, back-correction cap, velocity threshold
  ([`f0672e2`](https://github.com/tlancaster6/AquaPose/commit/f0672e212163a4e14bab2e26479b00dbc0346321))

- **06**: Capture phase context
  ([`26cbd0d`](https://github.com/tlancaster6/AquaPose/commit/26cbd0def2d53d7465eceaf41728821689e47a28))

- **06**: Create phase plan
  ([`5efe784`](https://github.com/tlancaster6/AquaPose/commit/5efe784f8c4a36c7317fcbccba9a809181a26233))

- **06**: Research 2D medial axis and arc-length sampling
  ([`2815491`](https://github.com/tlancaster6/AquaPose/commit/28154916b04d22b71a332cc09dcfbc3a8de739bd))

- **06-01**: Complete midline extraction plan — SUMMARY, STATE, ROADMAP, REQUIREMENTS updated
  ([`611bd50`](https://github.com/tlancaster6/AquaPose/commit/611bd50e066748fc5cd09bc8818f4cc5a53c1e06))

- **07**: Capture phase context
  ([`aca963b`](https://github.com/tlancaster6/AquaPose/commit/aca963b924902c6931cbbfb47181f997c301dfaa))

- **07**: Create phase plan
  ([`db87f72`](https://github.com/tlancaster6/AquaPose/commit/db87f72cf80a01458e700b47807453c1e6e78025))

- **07**: Research phase domain
  ([`1d0954e`](https://github.com/tlancaster6/AquaPose/commit/1d0954e8bc566072f777b367da864110dfe8a9c2))

- **07-01**: Complete multi-view triangulation plan summary and state
  ([`ab39308`](https://github.com/tlancaster6/AquaPose/commit/ab393087a8d1ec216f0821f38a32cbe0c1566cde))

- **08**: Capture phase context
  ([`ac7335a`](https://github.com/tlancaster6/AquaPose/commit/ac7335a5e6b55691337268f3cd5855de446370c9))

- **08**: Create phase plan
  ([`7d424ee`](https://github.com/tlancaster6/AquaPose/commit/7d424ee3edc69c0e75c69ddc63a71855a7469626))

- **08**: Research phase domain
  ([`6e54234`](https://github.com/tlancaster6/AquaPose/commit/6e542340313b03162aab81d3e17beb93f95ed7d1))

- **08-01**: Complete pipeline orchestrator plan execution
  ([`1a35437`](https://github.com/tlancaster6/AquaPose/commit/1a35437c7ead783410b44486f9d63598d3225756))

- **08-02**: Complete visualization and diagnostic report plan
  ([`cc21ce4`](https://github.com/tlancaster6/AquaPose/commit/cc21ce415672cf57f127b66722f1b8ec28099a0c))

- **08-03**: Update STATE.md — paused at checkpoint:human-verify Task 2
  ([`0664936`](https://github.com/tlancaster6/AquaPose/commit/06649365fc897356aae8ec771e8c741e5c507bb4))

- **09**: Capture phase context
  ([`4b12a31`](https://github.com/tlancaster6/AquaPose/commit/4b12a3128715e31f5e1d2d18029076ea2a34a608))

- **09**: Create phase plan for curve-based optimization
  ([`55e2b09`](https://github.com/tlancaster6/AquaPose/commit/55e2b09d46614f4b2a270d9a8972b16fc8afd01e))

- **09**: Research curve-based optimizer phase
  ([`617a92e`](https://github.com/tlancaster6/AquaPose/commit/617a92e2e40084195fa59f73d43d85f90ea2fa04))

- **09-01**: Complete CurveOptimizer plan - SUMMARY, STATE, ROADMAP updated
  ([`10bd07a`](https://github.com/tlancaster6/AquaPose/commit/10bd07a7f17e72556ccdbc166b2215e057813c64))

- **09-02**: Complete integration wiring plan - SUMMARY, STATE, ROADMAP updated
  ([`910dabe`](https://github.com/tlancaster6/AquaPose/commit/910dabe40350fa7328284563d5c4320a7c0015f6))

- **phase-01**: Complete phase verification and mark phase done
  ([`e35f4c5`](https://github.com/tlancaster6/AquaPose/commit/e35f4c511b393052ccc6e55cc8fbeed00ac042e7))

- **phase-02**: Add verification report (human_needed)
  ([`f36ab77`](https://github.com/tlancaster6/AquaPose/commit/f36ab778d85015657aa26df9d7f2a5943cd0c5f2))

- **phase-02.1.1**: Complete phase execution — YOLO detector wired into pipeline
  ([`32856d2`](https://github.com/tlancaster6/AquaPose/commit/32856d28e4cd528a49efd8bf62c71978720a3fc0))

- **phase-03**: Complete phase execution and verification
  ([`fe85af5`](https://github.com/tlancaster6/AquaPose/commit/fe85af5f59a9d31c525efea46bd1daeb7828d3dd))

- **phase-04.1**: Complete phase execution
  ([`13e1fa0`](https://github.com/tlancaster6/AquaPose/commit/13e1fa05e61c3042d59d9ffcc68f8fd2d48b6b0e))

- **phase-05**: Complete phase execution
  ([`868a2bf`](https://github.com/tlancaster6/AquaPose/commit/868a2bf709acfe9a6f0ef0deac1617219ff8d18a))

- **phase-06**: Complete phase execution and verification
  ([`cee154f`](https://github.com/tlancaster6/AquaPose/commit/cee154f2e5542b90166062fcc2bb6b6a1df5aa6f))

- **phase-07**: Complete phase execution and verification
  ([`1028c8d`](https://github.com/tlancaster6/AquaPose/commit/1028c8d7ed1cb6a10eeeb00d6044a5fbd3d1a267))

- **quick-1**: Fix triangulation bugs: NaN contamination, coupled thresholds
  ([`293bce2`](https://github.com/tlancaster6/AquaPose/commit/293bce2dc2334527bf6a4e0d34803b9bae08841b))

- **quick-2**: Create synthetic data module plan
  ([`4bc5a41`](https://github.com/tlancaster6/AquaPose/commit/4bc5a41f59b08baa64332ddb7c2f5f54573688b0))

- **quick-2-01**: Complete synthetic data module plan
  ([`239006c`](https://github.com/tlancaster6/AquaPose/commit/239006ce43eab519c499295ffc8420a87537fc95))

- **quick-3**: Add synthetic-mode diagnostic visualizations and report.md
  ([`d8da406`](https://github.com/tlancaster6/AquaPose/commit/d8da40663dedf18122fce884e6cacfa0a7f66a8d))

- **quick-3**: Complete add-synthetic-mode-diagnostic-visualizat plan
  ([`6f1f51f`](https://github.com/tlancaster6/AquaPose/commit/6f1f51fc3d62d41948cc24161bdde54d5a86c27c))

- **quick-4**: Add per-frame position and heading drift to synthetic fish
  ([`b77b29b`](https://github.com/tlancaster6/AquaPose/commit/b77b29b1bf836292340c63fe1d57065afaf61024))

- **quick-4**: Complete add-per-frame-position-drift-and-heading plan
  ([`9179b53`](https://github.com/tlancaster6/AquaPose/commit/9179b5374e34733698bd71e7fc315b8c5789d2af))

- **quick-4**: Sinusoidal shapes and diverse fish configs
  ([`46583b3`](https://github.com/tlancaster6/AquaPose/commit/46583b3034ca70788fe2d43403115a74055a2e93))

- **quick-5**: Add spline folding investigation report
  ([`cedbbe1`](https://github.com/tlancaster6/AquaPose/commit/cedbbe1bf6922af652913752c6aa14f1839820f3))

- **quick-5**: Complete spline-folding investigation plan
  ([`50b4754`](https://github.com/tlancaster6/AquaPose/commit/50b47549cc74fb2e877e437d8a674780b777b859))

- **quick-5**: Plan investigation of 3D spline folding and regularization
  ([`73e009d`](https://github.com/tlancaster6/AquaPose/commit/73e009d04ee0d8209a274d5668adaf6d799870df))

- **quick-5**: Update STATE.md with quick-5 completion
  ([`577d72d`](https://github.com/tlancaster6/AquaPose/commit/577d72d346eca6fa2a84dfb1f1e7586acbfe5850))

- **quick-6**: Complete synthetic data generation system plan
  ([`c31dd21`](https://github.com/tlancaster6/AquaPose/commit/c31dd21dbc8598e67b7869c5e6353a56b5406190))

- **quick-6**: Create synthetic data generation system plan
  ([`9f36aef`](https://github.com/tlancaster6/AquaPose/commit/9f36aef407fc83986b7eff66f4a19bc73c82e484))

- **quick-6**: Synthetic data generation system for tracker evaluation
  ([`d4a36bf`](https://github.com/tlancaster6/AquaPose/commit/d4a36bf83769c15cb85358e3b1d14903a021f0d9))

- **quick-7**: Complete FishTracker diagnostic plan
  ([`d81fcc2`](https://github.com/tlancaster6/AquaPose/commit/d81fcc22093980a82d27919b3a25c47eabd1f0ce))

- **quick-7**: Create cross-view identity and 3D tracking diagnostic plan
  ([`b310825`](https://github.com/tlancaster6/AquaPose/commit/b31082598e5f594debea0a3582bb17c201b772f5))

- **quick-7**: Verification and state update for tracking diagnostic
  ([`44f118c`](https://github.com/tlancaster6/AquaPose/commit/44f118ccaa37b023120a293999b3d4ba6149ebaa))

- **quick-8**: Complete windowed velocity smoothing plan
  ([`484a9a2`](https://github.com/tlancaster6/AquaPose/commit/484a9a2d4e20e524f1f11d6bcd418f45cfd17286))

- **quick-8**: Windowed velocity smoothing for tracking priors
  ([`c815370`](https://github.com/tlancaster6/AquaPose/commit/c8153701a209cff8880bb0ad986e7cf77606e083))

- **state**: Clarify 02.1.1-03 needs planning before execution
  ([`a8dbeab`](https://github.com/tlancaster6/AquaPose/commit/a8dbeab9f31601de2534b2f6a2dd8cf41ab278a1))

- **state**: Record phase 02.1 context session
  ([`18a4128`](https://github.com/tlancaster6/AquaPose/commit/18a4128e88a2fa5d0ca70015368a0f268fb9bc9e))

- **state**: Record phase 02.1.1 context session
  ([`a28bb16`](https://github.com/tlancaster6/AquaPose/commit/a28bb1661a6659e23c4059b39e43ea375817f686))

- **state**: Record phase 04.1 context session
  ([`a1355e9`](https://github.com/tlancaster6/AquaPose/commit/a1355e9f31239f31635eea654f48a09bd29d9a50))

- **state**: Record phase 1 context session
  ([`256041e`](https://github.com/tlancaster6/AquaPose/commit/256041e5eee974b6d692aa3c16631298cdd48b9f))

- **state**: Record phase 2 context session
  ([`77636ed`](https://github.com/tlancaster6/AquaPose/commit/77636ede025016ba77bfca07fb11c44446cf59de))

- **state**: Record phase 3 context session
  ([`56d9c5a`](https://github.com/tlancaster6/AquaPose/commit/56d9c5a86ed85148ed90501bb9927dc3cef9740e))

- **state**: Record phase 7 context session
  ([`ea14153`](https://github.com/tlancaster6/AquaPose/commit/ea14153ee9a1adb0a067eefa29b63e4085d8f1a8))

- **state**: Record phase 8 context session
  ([`57620e1`](https://github.com/tlancaster6/AquaPose/commit/57620e1c744acafca40026ea708e72bb170f4a94))

- **state**: Record phase 9 context session
  ([`32f9aab`](https://github.com/tlancaster6/AquaPose/commit/32f9aab439f4938605f1055a7257ade1ef8c92c4))

### Features

- Cross-camera flip alignment and strip dead orientation logic
  ([`3745bdf`](https://github.com/tlancaster6/AquaPose/commit/3745bdf499ab2c69a37b8f4caf6adb11194e6aa8))

- Diagnostic report, tracking fixes, and visualization overhaul
  ([`983525f`](https://github.com/tlancaster6/AquaPose/commit/983525f8e6bd2c2196bce5e47530e08578de1bc8))

- Improve training data sampling with spatial diversity and undistortion
  ([`c19330b`](https://github.com/tlancaster6/AquaPose/commit/c19330ba12e27a58ca91202c318ccdabad8de471))

- **01-01**: Add unit and cross-validation tests for calibration module
  ([`07f4416`](https://github.com/tlancaster6/AquaPose/commit/07f4416805b2a9057c939b6d796ddd0a7a26efd0))

- **01-01**: Port calibration loader and refractive projection model
  ([`6b46cb4`](https://github.com/tlancaster6/AquaPose/commit/6b46cb440b62618c44b7409ab63ae632cd931003))

- **01-02**: Generate Z-uncertainty characterization report
  ([`4ac597c`](https://github.com/tlancaster6/AquaPose/commit/4ac597c7504a6c86973b1f790115c1f84052d598))

- **01-02**: Implement Z-uncertainty characterization module
  ([`83e0b6c`](https://github.com/tlancaster6/AquaPose/commit/83e0b6caa18ba7ab9de99babe2eb4f4752fbb05c))

- **02-01**: Delete Label Studio module, move to_coco_dataset to pseudo_labeler
  ([`86529f5`](https://github.com/tlancaster6/AquaPose/commit/86529f5be2e32819e8ea8103270dacf743ec3503))

- **02-01**: Implement MOG2 fish detector with morphological cleanup
  ([`177b9bc`](https://github.com/tlancaster6/AquaPose/commit/177b9bc50f7d95555bbbf7c301927c120231b88b))

- **02-02**: Box-only SAM2 mode with quality filtering
  ([`003e8de`](https://github.com/tlancaster6/AquaPose/commit/003e8de3e0ba6e54dc9816134fd09f094e9736d3))

- **02-02**: Implement SAM2 pseudo-labeler and Label Studio IO
  ([`2b13849`](https://github.com/tlancaster6/AquaPose/commit/2b1384948e82b646f7a8b3c5ddb72bcae1530318))

- **02-02**: Variable-size CropDataset, stratified split, negative examples
  ([`2766e7d`](https://github.com/tlancaster6/AquaPose/commit/2766e7df4ea2ea26e0fb09a9b68b047831ab450b))

- **02-03**: Implement CropDataset and MaskRCNNSegmentor
  ([`69cef1f`](https://github.com/tlancaster6/AquaPose/commit/69cef1f76d5255db1addee5f35bb9e8c312573b1))

- **02-03**: Implement training script, evaluate, and pipeline API
  ([`b87fc08`](https://github.com/tlancaster6/AquaPose/commit/b87fc087fa5fa43d5001cf6fd746991bcdf39ebe))

- **02-03**: Refactor MaskRCNNSegmentor with segment() and crop-space output
  ([`5c4cbde`](https://github.com/tlancaster6/AquaPose/commit/5c4cbdeb6769e1db199b4bebf09e11b8660e6464))

- **02-03**: Update training pipeline for stratified split and variable crops
  ([`25ce169`](https://github.com/tlancaster6/AquaPose/commit/25ce1696fef25d48c7d940dce04f347e2768c9c4))

- **02-04**: Replace Mask R-CNN with lightweight U-Net segmentor
  ([`319490b`](https://github.com/tlancaster6/AquaPose/commit/319490b4010358a1e166bec338100bfce139345b))

- **02-04**: Wire updated components into build_training_data.py
  ([`3674e6b`](https://github.com/tlancaster6/AquaPose/commit/3674e6bbc4fcefcf1f36b79d5e0daa18d09870a8))

- **02.1-01**: Consolidate MOG2 diagnostic script with --cameras filter
  ([`3434adf`](https://github.com/tlancaster6/AquaPose/commit/3434adf676760e6688b1b2df6466498cc0ce3e17))

- **02.1-02**: Create SAM2 evaluation script for YOLO-sourced pseudo-labels
  ([`62252e6`](https://github.com/tlancaster6/AquaPose/commit/62252e6683dacdee7c5ecf80863e690f9641df91))

- **02.1-02**: Create SAM2 pseudo-label evaluation script
  ([`949d03f`](https://github.com/tlancaster6/AquaPose/commit/949d03faebbffa87bf56f4521393d4c2948d3dd7))

- **02.1.1-01**: Build MOG2-guided frame sampling script for YOLO annotation
  ([`3627a48`](https://github.com/tlancaster6/AquaPose/commit/3627a48d9c1a3fab3b38563166462c11684b0706))

- **02.1.1-01**: Organize annotated frames into YOLO train/val dataset
  ([`3fb9f45`](https://github.com/tlancaster6/AquaPose/commit/3fb9f45206003e5e1e3c17dbbbfaf8da81d69b94))

- **02.1.1-02**: Add YOLO training and eval scripts
  ([`e4596bf`](https://github.com/tlancaster6/AquaPose/commit/e4596bf2642a62f3febb6207f4c01e9404059107))

- **02.1.1-02**: Add YOLODetector and make_detector factory
  ([`41351ef`](https://github.com/tlancaster6/AquaPose/commit/41351efcdb521a16f671cb2298be29f8a915271e))

- **02.1.1-03**: Add end-to-end pseudo-label script and update docstrings
  ([`55a5310`](https://github.com/tlancaster6/AquaPose/commit/55a5310f5e100120164b08591c7a64e31024d583))

- **03-01**: FishState, profiles, spine, and cross-section modules with tests
  ([`d22ae89`](https://github.com/tlancaster6/AquaPose/commit/d22ae891dcb1fe93a4ae053e2e69edc1c24a6ec9))

- **03-01**: Mesh builder, PyTorch3D Meshes wrapping, and integration tests
  ([`40e4c0d`](https://github.com/tlancaster6/AquaPose/commit/40e4c0df15177a4909671c4d86beb6028c138f5a))

- **03-02**: Implement PCA keypoint extraction from binary masks
  ([`5122c9c`](https://github.com/tlancaster6/AquaPose/commit/5122c9c6f8b168231572ef938bc1a42bbff77029))

- **03-02**: Implement refractive triangulation and FishState initialization
  ([`d356a69`](https://github.com/tlancaster6/AquaPose/commit/d356a69f365f2e674bc98f0a438280350d056ae4))

- **04-01**: RefractiveCamera + RefractiveSilhouetteRenderer with tests
  ([`7284115`](https://github.com/tlancaster6/AquaPose/commit/728411566225f1583f009e8ca2bc77f7deb3ceda))

- **04-01**: Soft IoU loss, angular diversity weights, multi-objective loss
  ([`d474ec3`](https://github.com/tlancaster6/AquaPose/commit/d474ec3465d1fd75f7a81f0392a59dbb0b5b6217))

- **04-02**: Add unit tests for optimizer strategies (12 tests)
  ([`6e25b10`](https://github.com/tlancaster6/AquaPose/commit/6e25b107d7d7456e241e8c7aca078af32f44d59a))

- **04-02**: Implement FishOptimizer with 2-start and warm-start strategies
  ([`316868a`](https://github.com/tlancaster6/AquaPose/commit/316868a253cadeca904c30e1b154947ed1cb199f))

- **04-03**: Add end-to-end reconstruction CLI script
  ([`33b4deb`](https://github.com/tlancaster6/AquaPose/commit/33b4debd3f316634868f0739d2a91748ec156222))

- **04-03**: Implement holdout validation and visual overlay utilities
  ([`a37d592`](https://github.com/tlancaster6/AquaPose/commit/a37d5926dd66299041d7465ea9f04e846d04d905))

- **05-01**: Implement RANSAC centroid ray clustering for cross-view association
  ([`d546b71`](https://github.com/tlancaster6/AquaPose/commit/d546b7126a337bc3efa0f4e2718b6a70b96ff834))

- **05-02**: Implement FishTracker with Hungarian assignment and track lifecycle
  ([`226545e`](https://github.com/tlancaster6/AquaPose/commit/226545e073eaa4a80107d4beddcce1875ab1ba98))

- **05-03**: Implement HDF5 tracking writer with chunked datasets
  ([`ae33c38`](https://github.com/tlancaster6/AquaPose/commit/ae33c380317e0f816ca85dbea0f901a9cfb55d03))

- **06-01**: Add reconstruction package with Midline2D, MidlineExtractor, and pipeline helpers
  ([`50faf7a`](https://github.com/tlancaster6/AquaPose/commit/50faf7a1d1dc418b18f32684084641150689eb40))

- **07-01**: Implement multi-view triangulation module with Midline3D
  ([`bd83cc1`](https://github.com/tlancaster6/AquaPose/commit/bd83cc13993708cbc462063e8867fa482d41d774))

- **08-01**: Pipeline orchestrator, stage functions, and HDF5 midline writer
  ([`f85f02b`](https://github.com/tlancaster6/AquaPose/commit/f85f02b68e9dfa1c55b6f665ede63d91b650be79))

- **08-02**: 2D overlay renderer and 3D midline animation
  ([`be039b6`](https://github.com/tlancaster6/AquaPose/commit/be039b6b557fea9165b276bb49033d54e04dfa9b))

- **08-02**: Diagnostic report generator, orchestrator diagnostic mode, and overlay unit tests
  ([`f214b80`](https://github.com/tlancaster6/AquaPose/commit/f214b80e511e86c88364984e63805d44d7f89f9e))

- **08-03**: E2E integration test for full 5-stage reconstruction pipeline
  ([`0df6d96`](https://github.com/tlancaster6/AquaPose/commit/0df6d96cfccabfadf4a434979173c32ce126e136))

- **09-01**: Add unit tests for curve optimizer and fix acos NaN gradients
  ([`642eca4`](https://github.com/tlancaster6/AquaPose/commit/642eca4e92903882e92adb20bba0c0cde3af1ca0))

- **09-01**: Implement CurveOptimizer with coarse-to-fine B-spline optimization
  ([`4bd57eb`](https://github.com/tlancaster6/AquaPose/commit/4bd57ebc31f38d965c6feedb794fe9f574858925))

- **09-02**: Add --method flag to diagnose_pipeline.py for curve vs triangulation dispatch
  ([`8557f4b`](https://github.com/tlancaster6/AquaPose/commit/8557f4b5c936ffd25f17008a99bb7103485c9fc4))

- **quick-2-01**: Add synthetic data module for controlled pipeline testing
  ([`1bfee86`](https://github.com/tlancaster6/AquaPose/commit/1bfee862efb744c738f16ec6025c3f71ccb7a63e))

- **quick-2-01**: Add synthetic unit tests and --synthetic flag for diagnose_pipeline.py
  ([`b9dd734`](https://github.com/tlancaster6/AquaPose/commit/b9dd73447fb58161aceb0e6c7175323c483f07a8))

- **quick-3**: Add 4 synthetic diagnostic functions to diagnostics.py
  ([`19060e4`](https://github.com/tlancaster6/AquaPose/commit/19060e4259349e7bfd2a68fcd4519c97269b775c))

- **quick-3**: Wire 4 synthetic visualizations into _run_synthetic()
  ([`b3a4dd7`](https://github.com/tlancaster6/AquaPose/commit/b3a4dd7c88b3b877e4cb0446e3b0672a9a0d200e))

- **quick-4**: Add drift unit tests and update diagnose_pipeline defaults
  ([`48b12b5`](https://github.com/tlancaster6/AquaPose/commit/48b12b54a97fca367bf3ab9a4110d6b2e1994bf4))

- **quick-4**: Add sinusoidal spine shapes and diverse fish configs
  ([`58385c1`](https://github.com/tlancaster6/AquaPose/commit/58385c16d13fcdeb7ba6ec31849c4272dd670657))

- **quick-4**: Add velocity and angular_velocity drift fields to FishConfig
  ([`f4f9ae9`](https://github.com/tlancaster6/AquaPose/commit/f4f9ae94369152292feb12d38cd4947f98c8beae))

- **quick-6**: Detection generator projecting trajectories to noisy Detection objects
  ([`614c6b9`](https://github.com/tlancaster6/AquaPose/commit/614c6b9cfdbdc54bf8eea40de277490ad94716d7))

- **quick-6**: Scenario presets for targeted tracker failure mode evaluation
  ([`4c3c604`](https://github.com/tlancaster6/AquaPose/commit/4c3c604ccc8707d30932aaa7bbe28ee270267529))

- **quick-6**: Trajectory generator with heading-based random walk and social forces
  ([`3781536`](https://github.com/tlancaster6/AquaPose/commit/3781536831bf5d9519bf604593735688f6a46af4))

- **quick-7**: Cross-view identity and 3D tracking diagnostic script
  ([`0655121`](https://github.com/tlancaster6/AquaPose/commit/0655121e93806299d3ca7cc6cb3b2e666790a651))

- **quick-8**: Add velocity_history ring buffer and windowed smoothing to FishTrack
  ([`b73b123`](https://github.com/tlancaster6/AquaPose/commit/b73b123b796c9cc8269a40ba24902d504ab0c69c))

- **segmentation**: Add shared crop-segment-paste utilities
  ([`f2b03fe`](https://github.com/tlancaster6/AquaPose/commit/f2b03fe7fbd8a5d2d8096031e69e1bf55825dbd3))

- **tracking**: Add centroid video visualization and fix ghost track births
  ([`1fac02d`](https://github.com/tlancaster6/AquaPose/commit/1fac02da7380e04f710c67fd15b7e205ccb31ffa))

- **tracking**: Add tiled tracking video, improve detection and dedup
  ([`f9aeb75`](https://github.com/tlancaster6/AquaPose/commit/f9aeb759572a8f524dcb1bba52ade09159403e60))

- **tracking**: Replace Hungarian matching with track-driven association
  ([`7143584`](https://github.com/tlancaster6/AquaPose/commit/714358474d98ff2ff5dd9e9b4619fb8f80c4acd5))

### Refactoring

- Use VideoSet for undistortion in diagnostic visualizations
  ([`60d6bbf`](https://github.com/tlancaster6/AquaPose/commit/60d6bbf1d1017bbb524d76c4df30a3824fd0876a))

- **calibration**: Move triangulate_rays to projection, tidy post-phase cleanup
  ([`c9859f7`](https://github.com/tlancaster6/AquaPose/commit/c9859f7ba0f70b6dea4fed5488480b2b8ef0f6fe))

### Testing

- **02.1.1-03**: Add integration tests for YOLO -> SAM2 -> Label Studio pipeline
  ([`74c41ce`](https://github.com/tlancaster6/AquaPose/commit/74c41ce3bac012c95f733448bbf0b1d759fa0cc0))

- **05-01**: Add unit tests for RANSAC centroid clustering
  ([`04bc500`](https://github.com/tlancaster6/AquaPose/commit/04bc5006a27c0c2111dc599d08f40073ee34302e))

- **05-02**: Add unit tests for FishTracker lifecycle and Hungarian assignment
  ([`ad302b3`](https://github.com/tlancaster6/AquaPose/commit/ad302b326bc08c8289905fd8100e905e0d928923))

- **05-03**: Add unit tests for HDF5 writer round-trip and integration
  ([`57888c4`](https://github.com/tlancaster6/AquaPose/commit/57888c46a31d959a7ce64c97b8d38a072f8bd521))

- **06-01**: Add unit tests for midline extraction pipeline
  ([`0bb5b4d`](https://github.com/tlancaster6/AquaPose/commit/0bb5b4dc4612f2407988c41928b5dddca2ca5dc7))

- **07-01**: Add unit tests for multi-view triangulation module
  ([`e2b39a5`](https://github.com/tlancaster6/AquaPose/commit/e2b39a50e7ac1c4df45daab03db8d54c9f525adb))

- **08-01**: Unit tests for Midline3DWriter and pipeline stage functions
  ([`d0f4ff7`](https://github.com/tlancaster6/AquaPose/commit/d0f4ff730229b27b8170cd27fa9ebc0af703000b))

- **quick-7**: Smoke tests for diagnose_tracking metric computation
  ([`84a80ca`](https://github.com/tlancaster6/AquaPose/commit/84a80ca08f37199a6d55aee420055ad576c5eb55))

- **quick-8**: Add unit tests for windowed velocity smoothing
  ([`0cf5295`](https://github.com/tlancaster6/AquaPose/commit/0cf5295ac01ce23088de7946cd42b85c71223f1b))


## v1.0.0 (2026-02-19)

- Initial Release
