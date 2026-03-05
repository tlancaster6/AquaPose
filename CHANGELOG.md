# CHANGELOG

<!-- version list -->

## v1.1.0-dev.8 (2026-03-05)

### Bug Fixes

- Rename stale TestCarryForward test class to TestChunkHandoff
  ([`a3d7095`](https://github.com/tlancaster6/AquaPose/commit/a3d7095d8ef47485e4c5bedc45cca08baf2202bc))

- **54-03**: Fix typecheck errors in evaluation/viz modules
  ([`d5a3cd2`](https://github.com/tlancaster6/AquaPose/commit/d5a3cd2d2d49bf84911f7bda4f39c3fccf454540))

- **55-01**: Wire chunk_start through observer pipeline to fix manifest start_frame
  ([`355dd52`](https://github.com/tlancaster6/AquaPose/commit/355dd52aec78fa636ad0459fe2e135217c39d9cb))

- **assoc**: Pass undistortion maps when generating ForwardLUTs
  ([`f9a9bf3`](https://github.com/tlancaster6/AquaPose/commit/f9a9bf331a7b540c13253a0e0547e72251b2c927))

- **eval**: Exclude singletons from fish_yield_ratio, trim sweep grid
  ([`a671377`](https://github.com/tlancaster6/AquaPose/commit/a671377f9d08e6eb84de353b1a6400da98d01988))

- **reconstruction**: Iterative outlier rejection and spline extrapolation guard
  ([`937dd35`](https://github.com/tlancaster6/AquaPose/commit/937dd35ad11f06fd67172a9b9270c80b2a69284c))

- **tune**: Add legacy cache fallback, update grid test expectations
  ([`3d8ea7e`](https://github.com/tlancaster6/AquaPose/commit/3d8ea7e36d0864a8e9ee505662514b349de50800))

- **tune**: Return inf for empty centroid reprojection, tighten sweep grid
  ([`9e00bbb`](https://github.com/tlancaster6/AquaPose/commit/9e00bbb2151351dd7ea7c955c68c241810c18db9))

### Chores

- Complete v3.3 Chunk Mode milestone
  ([`b924daa`](https://github.com/tlancaster6/AquaPose/commit/b924daae39e367ce7bb8b87c26036064769a77d0))

- Complete v3.4 Performance Optimization milestone
  ([`9ee3e52`](https://github.com/tlancaster6/AquaPose/commit/9ee3e5286f1e896fac17d83103a23b8b84b0b556))

### Documentation

- Add Phase 60 End-to-End Performance Validation to roadmap
  ([`837cf6a`](https://github.com/tlancaster6/AquaPose/commit/837cf6a3267059fa2166c2e80101fd4d5201f1bd))

- Complete project research
  ([`e583833`](https://github.com/tlancaster6/AquaPose/commit/e583833540bc2cff7ca4f91e8c57ffb63d51a841))

- Create milestone v3.4 roadmap (4 phases)
  ([`968de59`](https://github.com/tlancaster6/AquaPose/commit/968de597c95657edd6eb3c19d3c7501077c48aeb))

- Define milestone v3.4 requirements
  ([`4374e22`](https://github.com/tlancaster6/AquaPose/commit/4374e2200b4ac4e8396a9448420983b004a040ff))

- Start milestone v3.4 Performance Optimization
  ([`35bf014`](https://github.com/tlancaster6/AquaPose/commit/35bf0142bda9c152ad47dd6a121836a4ba19a769))

- Update README for current architecture; minor type fixes
  ([`4673b4c`](https://github.com/tlancaster6/AquaPose/commit/4673b4c3707c765119094472eb5b21c06d5d8528))

- **54**: Capture phase context
  ([`8791d8c`](https://github.com/tlancaster6/AquaPose/commit/8791d8caef2bdd58af5fb42a97a2af0a32dae9a0))

- **54-01**: Complete chunk-aware diagnostic cache layout plan
  ([`44a39bc`](https://github.com/tlancaster6/AquaPose/commit/44a39bcc03711a45d9b2f5294f544e66783e7403))

- **54-02**: Add self-check result to SUMMARY.md
  ([`7ca4c64`](https://github.com/tlancaster6/AquaPose/commit/7ca4c64dc539682df539f26a6a23e533c117f16c))

- **54-02**: Complete eval migration to chunk cache layout plan
  ([`7403eb4`](https://github.com/tlancaster6/AquaPose/commit/7403eb4910afd7cb44929ed4e9756c6438b27578))

- **54-03**: Complete viz CLI subcommands plan
  ([`a33ae7e`](https://github.com/tlancaster6/AquaPose/commit/a33ae7e78459af4e262487eef177a8eef597803c))

- **54-04**: Complete remove viz observers plan
  ([`0d39fb3`](https://github.com/tlancaster6/AquaPose/commit/0d39fb325411a07bcf5f7a95db1df4d001c9b1ee))

- **55**: Capture phase context
  ([`a254cfe`](https://github.com/tlancaster6/AquaPose/commit/a254cfe691cd4d2026cc1ed4152d06594a941cc4))

- **55**: Create phase plan
  ([`6130e99`](https://github.com/tlancaster6/AquaPose/commit/6130e99fe7da634eaccf132a55464ece2e05167d))

- **55-01**: Complete chunk validation and gap closure plan
  ([`da155df`](https://github.com/tlancaster6/AquaPose/commit/da155df89d7963d98bfe4df194224e79b345d0ba))

- **56**: Capture phase context
  ([`aa1c2ee`](https://github.com/tlancaster6/AquaPose/commit/aa1c2ee179e9a4abe6126413ae8c693578064fc7))

- **56**: Complete phase research and planning
  ([`cff2eb7`](https://github.com/tlancaster6/AquaPose/commit/cff2eb7cf183d8d225e00c15be44c97dfce0c2e7))

- **56**: Research vectorized association scoring
  ([`c31ceb9`](https://github.com/tlancaster6/AquaPose/commit/c31ceb9bea11da77467dcbae252d3ed97617b802))

- **56-01**: Add plan summary
  ([`049be0f`](https://github.com/tlancaster6/AquaPose/commit/049be0fbe041b485a8b033f47f893a0cea321ddf))

- **56-02**: Add plan summary
  ([`50f7c99`](https://github.com/tlancaster6/AquaPose/commit/50f7c99ad555c4aec8049976645adbb6501d03ee))

- **57**: Capture phase context
  ([`343ba41`](https://github.com/tlancaster6/AquaPose/commit/343ba4108fae11bcfd83d452a4bac1d4bdc83009))

- **57**: Create phase plan
  ([`88dd2a8`](https://github.com/tlancaster6/AquaPose/commit/88dd2a8dbf925634a25574791b906ca64db97992))

- **57**: Research vectorized DLT reconstruction
  ([`e610c6c`](https://github.com/tlancaster6/AquaPose/commit/e610c6c39816d06cdaebf6915c2069763fd713a8))

- **57-01**: Complete vectorized DLT reconstruction plan
  ([`e4b573a`](https://github.com/tlancaster6/AquaPose/commit/e4b573a90c70464e482bbd8fbbe19fc08450a88b))

- **58**: Capture phase context
  ([`89c175d`](https://github.com/tlancaster6/AquaPose/commit/89c175dcba3832cff55ea1db7a03b6c19e213a9e))

- **58**: Create phase plan
  ([`4376769`](https://github.com/tlancaster6/AquaPose/commit/43767699089b40da2d61a7df20a1eea4ad6d9f00))

- **58**: Research frame I/O optimization phase
  ([`291f4ca`](https://github.com/tlancaster6/AquaPose/commit/291f4ca92d37376168a41fe96c46064ff71be67d))

- **58-01**: Complete frame I/O optimization plan
  ([`69934a3`](https://github.com/tlancaster6/AquaPose/commit/69934a3dec32f4b48c3e967c6b8aec9ea9f07702))

- **59**: Capture phase context
  ([`4c61948`](https://github.com/tlancaster6/AquaPose/commit/4c61948b7168af95e3e0492c68db68b9e1249f2e))

- **59**: Create phase plan
  ([`ffe8bd7`](https://github.com/tlancaster6/AquaPose/commit/ffe8bd78541d55d34350ef13be80bd142bbae418))

- **59**: Research phase domain
  ([`ea80c66`](https://github.com/tlancaster6/AquaPose/commit/ea80c664ae9657b2115db3e3ff49ea16b0d4018a))

- **59-01**: Complete OOM retry utility plan
  ([`096032c`](https://github.com/tlancaster6/AquaPose/commit/096032ca3a75e571a26cb7eaca1c5c29fe67e217))

- **59-02**: Complete batched detection backends plan
  ([`5a61e45`](https://github.com/tlancaster6/AquaPose/commit/5a61e459d02d9d3ac99e26f36cb21fb7b912a828))

- **59-03**: Complete batched midline inference plan
  ([`5e51d63`](https://github.com/tlancaster6/AquaPose/commit/5e51d63b3f2a143b4822fea0fe5db0ce58364c56))

- **60**: Capture phase context
  ([`8924852`](https://github.com/tlancaster6/AquaPose/commit/89248520abcb83b036c38060e18384d84afcd24f))

- **60**: Create phase plan for end-to-end performance validation
  ([`c09ae47`](https://github.com/tlancaster6/AquaPose/commit/c09ae47e6d24e664b0d7bb8c36ac541b9a0c494d))

- **60**: Research phase domain
  ([`9c866a9`](https://github.com/tlancaster6/AquaPose/commit/9c866a9fc278a4c981286920a1dd1e8063f5b735))

- **60-01**: Complete end-to-end performance validation plan
  ([`ad235e9`](https://github.com/tlancaster6/AquaPose/commit/ad235e9f7571124dc2e0fbe38ae905285386ee84))

- **60-01**: Generate v3.4 performance validation report
  ([`03129ad`](https://github.com/tlancaster6/AquaPose/commit/03129ade2ce86811d476aa0b2ad2ecbf1d8b51eb))

- **phase-54**: Complete phase execution
  ([`32261e5`](https://github.com/tlancaster6/AquaPose/commit/32261e58702be31f7711653bd855db4375d07897))

- **phase-55**: Complete phase execution
  ([`51147b4`](https://github.com/tlancaster6/AquaPose/commit/51147b48ba1c83c72cc63e3be68ed9e96870140b))

- **phase-56**: Complete phase execution
  ([`99d1009`](https://github.com/tlancaster6/AquaPose/commit/99d1009cc0d88a64623381c8ffa3869a7f644b1b))

- **phase-57**: Complete phase execution
  ([`8a0988e`](https://github.com/tlancaster6/AquaPose/commit/8a0988e941dace688e25424a690f39c09ebc892b))

- **phase-58**: Complete phase execution
  ([`afbcf4a`](https://github.com/tlancaster6/AquaPose/commit/afbcf4a07ea1507176cf40934fd86ac729403476))

- **phase-59**: Complete phase execution
  ([`c2a65ab`](https://github.com/tlancaster6/AquaPose/commit/c2a65ab5285b2e7642218e579c0004df61e79d15))

- **phase-60**: Complete phase execution
  ([`cf9a386`](https://github.com/tlancaster6/AquaPose/commit/cf9a38698e1fcf452306a020a15a042068876ace))

- **roadmap**: Add Phase 55 gap closure, fix chunk builder carry-over bug
  ([`38aa74b`](https://github.com/tlancaster6/AquaPose/commit/38aa74b212f30901b694b6468df24c1a5dfdc5fa))

### Features

- Add file-based debug logging for CLI commands
  ([`731d2c7`](https://github.com/tlancaster6/AquaPose/commit/731d2c78a3dc6f80e7d0a5637f8d1521f5f2d449))

- **54-01**: Remove diagnostic+chunk mutual exclusion, wire chunk_idx to observers
  ([`10647a8`](https://github.com/tlancaster6/AquaPose/commit/10647a884b60dbb398261e184d45a937098f3899))

- **54-01**: Restructure DiagnosticObserver for per-chunk single-cache layout
  ([`384089e`](https://github.com/tlancaster6/AquaPose/commit/384089e51dbc911a7019a974e84afe60ce3d3a80))

- **54-02**: Update EvalRunner for per-chunk cache layout
  ([`91c6732`](https://github.com/tlancaster6/AquaPose/commit/91c673295211306b9b9b8441fbe27c1aa8cf3ab0))

- **54-02**: Update TuningOrchestrator for chunk cache layout
  ([`c43536a`](https://github.com/tlancaster6/AquaPose/commit/c43536ad047c98a9816180b8f587eec1a2aa9e0f))

- **54-03**: Create evaluation/viz/ modules with overlay, animation, trails
  ([`8b69a4f`](https://github.com/tlancaster6/AquaPose/commit/8b69a4fb69000070fa3f7be2c8601c238d2ed1b7))

- **54-03**: Wire aquapose viz CLI group with overlay/animation/trails/all subcommands
  ([`b9369c1`](https://github.com/tlancaster6/AquaPose/commit/b9369c1f0949cb1f85c631165513af0bfdcbe243))

- **54-04**: Remove viz observers from engine, clean factory and init
  ([`4ca8102`](https://github.com/tlancaster6/AquaPose/commit/4ca8102a12247d32e0ba7820eb368c9b2dcca19f))

- **56-01**: Implement ray_ray_closest_point_batch with identity tests
  ([`9b4f3ae`](https://github.com/tlancaster6/AquaPose/commit/9b4f3aea1397f88ec22cfe7c1c8199c80f9b5229))

- **56-02**: Rewrite score_tracklet_pair with batched NumPy operations
  ([`a2427e2`](https://github.com/tlancaster6/AquaPose/commit/a2427e2581fa2658e826d9b81b9017c6bafdc689))

- **57-01**: Implement _TriangulationResult and _triangulate_fish_vectorized()
  ([`cd3b68a`](https://github.com/tlancaster6/AquaPose/commit/cd3b68a09d487476f53c302186ec85db03a52790))

- **57-01**: Wire _reconstruct_fish() to vectorized path and add equivalence tests
  ([`3ed2527`](https://github.com/tlancaster6/AquaPose/commit/3ed25274fee777965ca3b21aa45cea27fb040ffd))

- **58-01**: Implement background prefetch in ChunkFrameSource
  ([`e8a95da`](https://github.com/tlancaster6/AquaPose/commit/e8a95da17b7db00501660cb5c179829e3c4a06c2))

- **59-01**: Add batch size config fields to DetectionConfig and MidlineConfig
  ([`32a1903`](https://github.com/tlancaster6/AquaPose/commit/32a1903e471073ab0e0b902aad331bc91eb7311d))

- **59-01**: Add BatchState and predict_with_oom_retry utility
  ([`f768f9b`](https://github.com/tlancaster6/AquaPose/commit/f768f9b6dde7e0ea83d693676953faa28b80e1e1))

- **59-02**: Add detect_batch() to YOLOOBBBackend and YOLOBackend
  ([`55575b0`](https://github.com/tlancaster6/AquaPose/commit/55575b01906bc2bf11a601e4b2dffeacfee42db5))

- **59-02**: Refactor DetectionStage.run() for batched inference
  ([`375ad95`](https://github.com/tlancaster6/AquaPose/commit/375ad95e0256a503bcb3a75cf6469c7341358b8b))

- **59-03**: Add process_batch() to both midline backends
  ([`8fb3345`](https://github.com/tlancaster6/AquaPose/commit/8fb33456d8f976d3604b3b8730714f0d2e4476af))

- **59-03**: Refactor MidlineStage.run() for batched midline inference
  ([`94bf767`](https://github.com/tlancaster6/AquaPose/commit/94bf767af94b9307de713956732f9373eb0488c4))

- **60-01**: Add performance validation script
  ([`b650319`](https://github.com/tlancaster6/AquaPose/commit/b650319ef119f4ded338e6297fdf517c4a4be52f))

### Refactoring

- **cli**: Flatten viz subcommands into flag-based single command
  ([`24018cb`](https://github.com/tlancaster6/AquaPose/commit/24018cb977e42c3e030f149af6eadcf9c2795d76))

- **tune**: Replace full reconstruction with centroid reprojection
  ([`bc89a82`](https://github.com/tlancaster6/AquaPose/commit/bc89a8255b48e0307583b868f03aae2526e6d444))

- **viz**: Delete dead visualization code, optimize overlay and trails
  ([`63fa4f9`](https://github.com/tlancaster6/AquaPose/commit/63fa4f9daaf360e4bd0321712861b00cc9a71586))

### Testing

- **54-01**: Add failing tests for per-chunk single-cache layout
  ([`0e0c98d`](https://github.com/tlancaster6/AquaPose/commit/0e0c98d377578b73e22d1ecdd442ff89852bd7fa))

- **55-01**: Add INTEG-03 validation tests and update REQUIREMENTS.md
  ([`aa79856`](https://github.com/tlancaster6/AquaPose/commit/aa79856ad925592e2708d898d30be7db807d8e96))

- **56-02**: Add early termination edge case tests for batched scoring
  ([`d1b7699`](https://github.com/tlancaster6/AquaPose/commit/d1b76995b6d445d1f40514a6a962b5037cc5d4d6))

- **58-01**: Add failing prefetch tests for ChunkFrameSource
  ([`2c0ab6e`](https://github.com/tlancaster6/AquaPose/commit/2c0ab6e35e9f43abefa83998da7799b2cdcd7f9c))


## v1.1.0-dev.7 (2026-03-04)

### Bug Fixes

- Edge extrapolation overshoot and multi-fish pose annotations
  ([`517b017`](https://github.com/tlancaster6/AquaPose/commit/517b017ddf94ac74f1c0497c01d344539353c804))

- **47-02**: Fix import order in test_stage_midline.py
  ([`6280f60`](https://github.com/tlancaster6/AquaPose/commit/6280f607f3a3d682d165cc9642dd180ccfe9b313))

- **52-02**: Fix typecheck errors in orchestrator.py
  ([`1e239a8`](https://github.com/tlancaster6/AquaPose/commit/1e239a84235447b2252a64eb7d832e9fc928999e))

- **52-03**: Update tests and ocsort_wrapper to use ChunkHandoff
  ([`041e556`](https://github.com/tlancaster6/AquaPose/commit/041e556b7aeb5c78437418c3d76bddaa337591fb))

- **53-01**: Don't pre-create VideoFrameSource in CLI — let orchestrator manage lifecycle
  ([`123ccb8`](https://github.com/tlancaster6/AquaPose/commit/123ccb814889ced43116b7a9f900dfad92aae576))

- **viz**: Improve overlay mosaic readability
  ([`aeb3250`](https://github.com/tlancaster6/AquaPose/commit/aeb32500990ed087cb0b26db75f4b7f7057dc5b4))

### Chores

- Clean up v3.2 tech debt — stale docs, legacy golden/regression tests
  ([`50c25a2`](https://github.com/tlancaster6/AquaPose/commit/50c25a2405d21ca198aeaf695de17cd6b63f630c))

- Complete v3.2 Evaluation Ecosystem milestone
  ([`200d0c0`](https://github.com/tlancaster6/AquaPose/commit/200d0c08876f2a16475112da59c54dbe92f07740))

- Triage pending TODOs for v3.2 milestone launch
  ([`b994a7a`](https://github.com/tlancaster6/AquaPose/commit/b994a7a3ff12000b85caf18c4b9b6f14dfe373a9))

- **49-02**: Delete legacy tuning scripts
  ([`c17312d`](https://github.com/tlancaster6/AquaPose/commit/c17312d2a1ecc315badf501acc61b1f1ea7a1e5b))

- **50-01**: Clean up legacy tests and fix docstring references
  ([`fc03147`](https://github.com/tlancaster6/AquaPose/commit/fc031475ce518cdd0520af77fce096ce0646c349))

- **50-01**: Delete legacy evaluation machinery and midline fixture NPZ
  ([`ba47aa8`](https://github.com/tlancaster6/AquaPose/commit/ba47aa8c85202658bfcae78b1671d764f7086d67))

### Documentation

- Add GPU PyTorch install instructions to README
  ([`13283ff`](https://github.com/tlancaster6/AquaPose/commit/13283ffcf78c78f071840943abe47c9b65aa3f13))

- Capture todo - Migrate visualization observers to eval suite
  ([`a33c89b`](https://github.com/tlancaster6/AquaPose/commit/a33c89b6fe8b74fc99c59d53a5605fd36b583fda))

- Complete project research
  ([`65bdd3f`](https://github.com/tlancaster6/AquaPose/commit/65bdd3fe92e89550a0aa9eb620e115c763fa62e7))

- Create milestone v3.2 roadmap (5 phases)
  ([`eceba29`](https://github.com/tlancaster6/AquaPose/commit/eceba29d0b65c58d05faadd3b8b527169e399212))

- Create milestone v3.3 roadmap (3 phases)
  ([`5136782`](https://github.com/tlancaster6/AquaPose/commit/513678276ce1e2d07bbdb5a8db8af33b2af2b512))

- Define milestone v3.2 requirements
  ([`0a6acfd`](https://github.com/tlancaster6/AquaPose/commit/0a6acfd5d917406c515ca99980571bea247f773a))

- Define milestone v3.3 requirements
  ([`24484b0`](https://github.com/tlancaster6/AquaPose/commit/24484b0298ca8511d717185433a6a7bf38f552d1))

- Refine evaluation system design for v3.2
  ([`63232b9`](https://github.com/tlancaster6/AquaPose/commit/63232b969c5891fe607f5f0aca23b53db48931cf))

- Start milestone v3.2 Evaluation Ecosystem
  ([`1f7f0c2`](https://github.com/tlancaster6/AquaPose/commit/1f7f0c21576765d0a092043d3de1c07f9ff6539d))

- Start milestone v3.3 Chunk Mode
  ([`329ffbb`](https://github.com/tlancaster6/AquaPose/commit/329ffbb5edfb4523bfbe361b4ae728266b7346ba))

- Update planning state — phases 51-52 complete, add phase 54, gitignore .pt files
  ([`3e2dbd8`](https://github.com/tlancaster6/AquaPose/commit/3e2dbd81044da051c87bda82ec8b13a1a8f847a8))

- **46**: Capture phase context
  ([`cccffb8`](https://github.com/tlancaster6/AquaPose/commit/cccffb89eaf564daa965f8df92e8ab405c7a9940))

- **46**: Capture phase research and plans
  ([`58a1490`](https://github.com/tlancaster6/AquaPose/commit/58a1490b59d6072c9bb98aa24ca76b85cf20a82f))

- **46**: Fix plan file naming convention for GSD tooling
  ([`63c7197`](https://github.com/tlancaster6/AquaPose/commit/63c719701aaf22d2efe51a480125c64763dbb80d))

- **46**: Research phase domain
  ([`7253575`](https://github.com/tlancaster6/AquaPose/commit/725357504b5d3e3a1c303b8c4c614c1aececefa4))

- **46-01**: Complete StaleCacheError, load_stage_cache, context_fingerprint plan
  ([`7ac4a78`](https://github.com/tlancaster6/AquaPose/commit/7ac4a785ff3c59e96192b6555251983098663fcc))

- **46-02**: Complete per-stage pickle cache writing and stage-skip plan
  ([`061efc1`](https://github.com/tlancaster6/AquaPose/commit/061efc1e968c30f8270602b8c4630b55bd05943f))

- **46-03**: Complete CLI --resume-from plan with summary and state update
  ([`92e691b`](https://github.com/tlancaster6/AquaPose/commit/92e691b7edb9e1d043dcf376e761783b49c14124))

- **47**: Add phase plans (3 plans, 2 waves)
  ([`e38ef4f`](https://github.com/tlancaster6/AquaPose/commit/e38ef4f9dd3b9d70d0458245e767737ac7e55b7d))

- **47**: Capture phase context
  ([`188d127`](https://github.com/tlancaster6/AquaPose/commit/188d1273be4cc7be42a3d383105a0705c14f42ab))

- **47**: Research evaluation primitives phase
  ([`0d2392c`](https://github.com/tlancaster6/AquaPose/commit/0d2392c5472181cde0fdbdcdec097f9fd2423dbf))

- **47-01**: Complete detection and tracking stage evaluators plan with summary and state update
  ([`2fd9f5b`](https://github.com/tlancaster6/AquaPose/commit/2fd9f5b1af20c62188f34b9c5d59c774cbd5ebd4))

- **47-02**: Complete association and midline stage evaluators plan
  ([`d15ea3d`](https://github.com/tlancaster6/AquaPose/commit/d15ea3dad73b21c4c1cbb58f9109b315de51aebf))

- **47-03**: Complete reconstruction stage evaluator plan with summary and state update
  ([`14be136`](https://github.com/tlancaster6/AquaPose/commit/14be1368a73bfa8e49763ac58ea87695e9871c42))

- **48**: Capture phase context
  ([`d372fb7`](https://github.com/tlancaster6/AquaPose/commit/d372fb77fb883293025f470a9d8d15bca477b1c6))

- **48**: Create phase plan
  ([`d1c55c8`](https://github.com/tlancaster6/AquaPose/commit/d1c55c8c3e54dac0afef9ff07da3a9b6854a1b78))

- **48**: Fix verify command to use hatch run python
  ([`4d0dd6a`](https://github.com/tlancaster6/AquaPose/commit/4d0dd6a23c11c0bc3301f7f330617127b8daa23c))

- **48**: Research phase domain
  ([`12e7c00`](https://github.com/tlancaster6/AquaPose/commit/12e7c004e9ddbc7dc544364cb25fd374ec9a465a))

- **48-01**: Complete EvalRunner plan with summary and state update
  ([`f71b769`](https://github.com/tlancaster6/AquaPose/commit/f71b7691f195aad95ad9eb806e435ae0cdddb28d))

- **48-02**: Complete output formatters and eval CLI plan with summary and state update
  ([`791acc5`](https://github.com/tlancaster6/AquaPose/commit/791acc58a22641abe4caaa2069635f7340ea4b3a))

- **49**: Capture phase context
  ([`c49b06c`](https://github.com/tlancaster6/AquaPose/commit/c49b06cc8ae9bd5065aa0896ef0e3900e5942604))

- **49**: Create phase plan
  ([`3119606`](https://github.com/tlancaster6/AquaPose/commit/31196068ea92c43d2139dfc134cd656593a7f48d))

- **49**: Research phase domain
  ([`739fe03`](https://github.com/tlancaster6/AquaPose/commit/739fe0302dfb5e77db6d25d2fdf1939f99ec6185))

- **49-01**: Complete TuningOrchestrator plan with summary and state update
  ([`497c478`](https://github.com/tlancaster6/AquaPose/commit/497c478eb5b7c19b138ca4cb27c6e3178b98ebd5))

- **49-02**: Complete aquapose tune CLI plan with summary and state update
  ([`e1d27fa`](https://github.com/tlancaster6/AquaPose/commit/e1d27fa53b260f338c7c03a8b2c8da75254eacda))

- **50**: Capture phase context
  ([`65a23d5`](https://github.com/tlancaster6/AquaPose/commit/65a23d5a59608b0471e82f3f0c1a12af6f2551e6))

- **50**: Create phase plan
  ([`87c3e94`](https://github.com/tlancaster6/AquaPose/commit/87c3e948797ad3f1bcca7aea419e731c15c862c9))

- **50**: Research phase domain
  ([`5a578f2`](https://github.com/tlancaster6/AquaPose/commit/5a578f2646b54fb87e6df25f64e6c1e0cda8e7ca))

- **50-01**: Complete legacy evaluation cleanup plan with summary and state update
  ([`3caec62`](https://github.com/tlancaster6/AquaPose/commit/3caec627d212bfa81842d9af4eb6dc422edbfb5b))

- **51**: Capture phase context
  ([`202ef5d`](https://github.com/tlancaster6/AquaPose/commit/202ef5df22bd8f031e5e66043383593e5b3c9779))

- **51**: Create phase plan
  ([`f3ba761`](https://github.com/tlancaster6/AquaPose/commit/f3ba76157182edcadea3d7c416924e69c9656e1e))

- **51-01**: Complete frame-source-refactor plan 01
  ([`f1eb790`](https://github.com/tlancaster6/AquaPose/commit/f1eb790f3976fa8c4339edf100f429cffedbec24))

- **51-02**: Complete observer-migration and VideoSet-deletion plan
  ([`1385d20`](https://github.com/tlancaster6/AquaPose/commit/1385d204ea9fe6c70461da078c1881c29fec7278))

- **52**: Capture phase context
  ([`544a3aa`](https://github.com/tlancaster6/AquaPose/commit/544a3aae245a4cad0b6120136512eb4120b20df9))

- **52**: Create phase plans for chunk orchestrator and handoff
  ([`633788e`](https://github.com/tlancaster6/AquaPose/commit/633788eec649a3c8b4911cd3b58e9699bff4a005))

- **52-01**: Complete chunk orchestrator foundation plan
  ([`8858536`](https://github.com/tlancaster6/AquaPose/commit/885853680638742d23d0aec2e6106e6dc222e2c7))

- **52-02**: Complete ChunkOrchestrator plan
  ([`0a4acc2`](https://github.com/tlancaster6/AquaPose/commit/0a4acc2f34c77a091b1968220604f52a4243ff9d))

- **52-03**: Complete CarryForward migration plan — Phase 52 all requirements met
  ([`4f8992e`](https://github.com/tlancaster6/AquaPose/commit/4f8992e67b4a31a3baf2abdb2f6c0a3fe0ed0068))

- **53**: Capture phase context
  ([`d3c0874`](https://github.com/tlancaster6/AquaPose/commit/d3c0874d4e72e2386e4d818756033cd0962528b5))

- **53-01**: Complete ChunkOrchestrator integration plan — OUT-02, INTEG-01, INTEG-02 met
  ([`b9affea`](https://github.com/tlancaster6/AquaPose/commit/b9affea57853d93be15340d3dead9c34743b3005))

- **phase-46**: Complete phase execution
  ([`a547b98`](https://github.com/tlancaster6/AquaPose/commit/a547b988aba29ffb6f84fdbdbf735ff208596e0e))

- **phase-47**: Complete phase execution
  ([`acf06cf`](https://github.com/tlancaster6/AquaPose/commit/acf06cf8a4ad7ecb4720f46fc62d6f1c35858e6b))

- **phase-48**: Complete phase execution
  ([`a86c080`](https://github.com/tlancaster6/AquaPose/commit/a86c080cda373ba0795a7958eac3f0a41a16ce7b))

- **phase-49**: Complete phase execution
  ([`b53e5a2`](https://github.com/tlancaster6/AquaPose/commit/b53e5a26b45604931a6943d9375081d7efdae7d8))

- **phase-50**: Complete phase execution
  ([`9f3ddda`](https://github.com/tlancaster6/AquaPose/commit/9f3ddda3733764bf127c1e4f7cc06a3f39f86fef))

- **phase-51**: Complete phase execution
  ([`5ad9639`](https://github.com/tlancaster6/AquaPose/commit/5ad96392e36e08a292ded67f1f3d5ef421730f91))

- **phase-52**: Complete phase execution
  ([`c83a868`](https://github.com/tlancaster6/AquaPose/commit/c83a868220957d56f32debf981bb66ac559e6368))

### Features

- Add --mosaic flag to YOLO training CLI commands
  ([`2c50f8e`](https://github.com/tlancaster6/AquaPose/commit/2c50f8e7faac3e5fc866898a29cad69d7d3fe3a6))

- Add --patience option to train CLI subcommands
  ([`14b0b5d`](https://github.com/tlancaster6/AquaPose/commit/14b0b5d84b92672d23f0087ae145059f1201556c))

- Color singletons gray in tracklet trail observer
  ([`fcf10b0`](https://github.com/tlancaster6/AquaPose/commit/fcf10b08e7f5f915ef800b814368660fbe03a764))

- **46-01**: Add carry_forward field, StaleCacheError, context_fingerprint, and load_stage_cache to
  context.py
  ([`9216797`](https://github.com/tlancaster6/AquaPose/commit/921679738d87c18b3dca5db839609f48cc2b3ec0))

- **46-01**: Export StaleCacheError, load_stage_cache, context_fingerprint from core.__init__ and
  add unit tests
  ([`339e769`](https://github.com/tlancaster6/AquaPose/commit/339e769943e3479cae306686a1dc69cfeae7f282))

- **46-02**: Add initial_context, stage-skip logic, and cache write tests
  ([`3d732b6`](https://github.com/tlancaster6/AquaPose/commit/3d732b67109da69e7b8ecbe0d233754c3ff8392b))

- **46-03**: Add --resume-from CLI flag with integration tests
  ([`a9e3e1b`](https://github.com/tlancaster6/AquaPose/commit/a9e3e1b26204544b6448fd28b68dee7adbe1bb4f))

- **47-01**: Add evaluation/stages/ subpackage with detection evaluator
  ([`08e3791`](https://github.com/tlancaster6/AquaPose/commit/08e3791fee4ef36a590f04ab2f7a14b37a636065))

- **47-01**: Add tracking evaluator with TrackingMetrics and evaluate_tracking()
  ([`b104d16`](https://github.com/tlancaster6/AquaPose/commit/b104d162a9d1b104e095a7f2f2be308d5d89c808))

- **47-02**: Implement AssociationMetrics, evaluate_association(), and DEFAULT_GRID
  ([`8fcf8c0`](https://github.com/tlancaster6/AquaPose/commit/8fcf8c0548a2821e83eb6e5a6c9dafe8a470ef1b))

- **47-02**: Implement MidlineMetrics and evaluate_midline()
  ([`c69254b`](https://github.com/tlancaster6/AquaPose/commit/c69254b33906e8ddafb6d82df1647cd908191e9f))

- **47-03**: Implement reconstruction stage evaluator
  ([`0d57e98`](https://github.com/tlancaster6/AquaPose/commit/0d57e98d1f62f2dd3b8163497a3ca8d491f41407))

- **47-03**: Wire __init__.py exports for stages/ and evaluation/
  ([`9692f71`](https://github.com/tlancaster6/AquaPose/commit/9692f71bb4a2c6a29f7fa7e44c4030a7cfcc90c5))

- **48-01**: Implement EvalRunner and EvalRunnerResult for per-stage evaluation
  ([`bbae5a6`](https://github.com/tlancaster6/AquaPose/commit/bbae5a661c544014b12d4d9d505f1ed09315b39a))

- **48-02**: Add aquapose eval CLI command, update __init__.py, delete measure_baseline.py
  ([`053a653`](https://github.com/tlancaster6/AquaPose/commit/053a6538f763611f5edf5682fb2942de25597520))

- **48-02**: Add format_eval_report and format_eval_json to output.py
  ([`c93fc82`](https://github.com/tlancaster6/AquaPose/commit/c93fc8235e3e637ca49a767a0e656a4550ea403c))

- **49-01**: Implement TuningOrchestrator with grid sweep engine
  ([`16d167f`](https://github.com/tlancaster6/AquaPose/commit/16d167f5a53f91a90cc8c818e56a05eba3f3273b))

- **49-02**: Add aquapose tune CLI command
  ([`dd00bc3`](https://github.com/tlancaster6/AquaPose/commit/dd00bc324cdac656a9bd49531a285a14ec9aa071))

- **51-01**: Define FrameSource protocol and VideoFrameSource implementation
  ([`b89eb4e`](https://github.com/tlancaster6/AquaPose/commit/b89eb4e58a561e77ff4444f8faa875abbf6e9187))

- **51-01**: Migrate DetectionStage and MidlineStage to accept FrameSource
  ([`1357b68`](https://github.com/tlancaster6/AquaPose/commit/1357b6801a4bd88415af8e0a6b933894b94d53b3))

- **51-02**: Migrate observers to FrameSource, update observer_factory and CLI
  ([`c23abf8`](https://github.com/tlancaster6/AquaPose/commit/c23abf875977c53ca75365a70f215ddb603ae5d5))

- **51-02**: Remove stop_frame from config, delete VideoSet, update tests
  ([`64ba5cc`](https://github.com/tlancaster6/AquaPose/commit/64ba5cc4482b223d14125c46d4c0c554f82520d3))

- **52-01**: Add chunk_size field to PipelineConfig with warning
  ([`ed304f6`](https://github.com/tlancaster6/AquaPose/commit/ed304f64ac8206066c266d9c784a8b70649cf574))

- **52-01**: Create engine/orchestrator.py with ChunkHandoff and write_handoff
  ([`777d4e8`](https://github.com/tlancaster6/AquaPose/commit/777d4e868d8e84e049d8a28c57025c9735355a7b))

- **52-01**: Implement ChunkFrameSource windowed frame source
  ([`d1093af`](https://github.com/tlancaster6/AquaPose/commit/d1093af467604103f16f9878660688b3d8f06412))

- **52-02**: Add track_id_to_global to ChunkHandoff and _stitch_identities helper
  ([`24e8b4a`](https://github.com/tlancaster6/AquaPose/commit/24e8b4a77f302ea3ae2b6ba01c896c2a4067c7d7))

- **52-02**: Export ChunkOrchestrator and add unit tests
  ([`a696e9b`](https://github.com/tlancaster6/AquaPose/commit/a696e9b0b79dca2c08aedf42795b3d9676a6976f))

- **52-02**: Implement ChunkOrchestrator with run(), helper functions
  ([`bf91a9d`](https://github.com/tlancaster6/AquaPose/commit/bf91a9dffaf4b0eda68411fb0ec3c0044d102c89))

- **53-01**: Add max_chunks, stop_after, extra_observers to ChunkOrchestrator
  ([`db5ebef`](https://github.com/tlancaster6/AquaPose/commit/db5ebef678261369d4030297a44b06c83e67b4fd))

- **53-01**: Delete HDF5ExportObserver and remove all references
  ([`96187c9`](https://github.com/tlancaster6/AquaPose/commit/96187c987c48a5ec28d74abcb9e93ec1a5b6866e))

- **53-01**: Refactor CLI run command to delegate to ChunkOrchestrator
  ([`9fc6fef`](https://github.com/tlancaster6/AquaPose/commit/9fc6fef68c24af4c66ff00a020f8511f9c4e1cbd))

- **53-01**: Update tests for new CLI dispatch and mode conflict validation
  ([`bf6aa49`](https://github.com/tlancaster6/AquaPose/commit/bf6aa49c097dd72498e01c6f96096687bfae535f))

### Refactoring

- **52-03**: Move ChunkHandoff to core/context.py, remove CarryForward
  ([`fbb0fd2`](https://github.com/tlancaster6/AquaPose/commit/fbb0fd27e86d3f730cb3b9286437f3e9f12d5e80))

- **52-03**: Update TrackingStage and ChunkOrchestrator to use ChunkHandoff
  ([`d6e50d4`](https://github.com/tlancaster6/AquaPose/commit/d6e50d4338c43f10c407c9d178218f5629ee44cc))

### Testing

- **47-02**: Add failing tests for association evaluator and DEFAULT_GRID
  ([`6a19fc7`](https://github.com/tlancaster6/AquaPose/commit/6a19fc7c02033d49997c98dab89e9bbbfabb1ddf))

- **47-02**: Add failing tests for MidlineMetrics and evaluate_midline()
  ([`f27ade7`](https://github.com/tlancaster6/AquaPose/commit/f27ade72ca64444aa8783bce0d4df29e741bcd1a))

- **47-03**: Add failing tests for reconstruction stage evaluator
  ([`c0e914f`](https://github.com/tlancaster6/AquaPose/commit/c0e914f40b3862f462a8c0ae21a42929375457fe))

- **48-01**: Add failing tests for EvalRunner with synthetic cache fixtures
  ([`e547e9b`](https://github.com/tlancaster6/AquaPose/commit/e547e9bb14141c6c55d96adee94e4b7329c05713))

- **48-02**: Add failing tests for format_eval_report and format_eval_json
  ([`9bfbf7a`](https://github.com/tlancaster6/AquaPose/commit/9bfbf7aad9183263aedeb37e476ffb0169670e69))

- **51-01**: Add failing tests for FrameSource protocol and VideoFrameSource
  ([`7899062`](https://github.com/tlancaster6/AquaPose/commit/78990622fb64f09e0147ea3e805c325a1004e10d))

- **52-01**: Add unit tests for ChunkHandoff, ChunkFrameSource, write_handoff
  ([`82e2ca1`](https://github.com/tlancaster6/AquaPose/commit/82e2ca120f4f93dae085bf87b4c847e90f6188b4))


## v1.1.0-dev.6 (2026-03-03)

### Bug Fixes

- Wire calibration into DiagnosticObserver for v2.0 fixture export
  ([`e93cf8d`](https://github.com/tlancaster6/AquaPose/commit/e93cf8d07abb61d3794d27571ff0f3497cad2f58))

### Chores

- Complete v3.1 Reconstruction milestone
  ([`92e3eb8`](https://github.com/tlancaster6/AquaPose/commit/92e3eb86a2759dc2ab6418e536a4fb7e1636525f))

- Remove phase 46 (evaluation-unification-and-optimization)
  ([`b9e56c9`](https://github.com/tlancaster6/AquaPose/commit/b9e56c9b7aefbbb183f3dd8e47666df7d00d6c09))

### Documentation

- Create milestone v3.1 roadmap (6 phases)
  ([`d9efb3d`](https://github.com/tlancaster6/AquaPose/commit/d9efb3d40dff44c128c6dcd32394c191a3dab401))

- Define milestone v3.1 requirements
  ([`0b92fd6`](https://github.com/tlancaster6/AquaPose/commit/0b92fd65ac233771a5eb9f9182a763fee4a9f6e2))

- Start milestone v3.1 Reconstruction
  ([`14f7eec`](https://github.com/tlancaster6/AquaPose/commit/14f7eecf19e18013649a604f9f6c4847776d92d4))

- **40**: Capture phase context
  ([`993e133`](https://github.com/tlancaster6/AquaPose/commit/993e13373811197bfe594b928bd0b5d51c87ac40))

- **40-01**: Complete midline fixture serialization plan
  ([`3479982`](https://github.com/tlancaster6/AquaPose/commit/3479982ac12da1bc97411fdbb9c7baf2f3fe5451))

- **40-02**: Complete midline fixture loader plan
  ([`bc094df`](https://github.com/tlancaster6/AquaPose/commit/bc094df6ca63b6cf0495fd9c3e6f59eb53631836))

- **41**: Create evaluation harness phase plan
  ([`9eb4719`](https://github.com/tlancaster6/AquaPose/commit/9eb4719233f2e5f939852a2fd6cd3289284ebb97))

- **41**: Research evaluation harness phase domain
  ([`060ab5d`](https://github.com/tlancaster6/AquaPose/commit/060ab5d27fb93234ba9943a5674a740ce592708b))

- **41-01**: Complete CalibBundle and NPZ v2.0 calibration bundling plan
  ([`7bf02ef`](https://github.com/tlancaster6/AquaPose/commit/7bf02ef622a342ea0b57d5187eb057d76016769d))

- **41-02**: Complete evaluation harness plan
  ([`b653232`](https://github.com/tlancaster6/AquaPose/commit/b653232978422993a84226cd9e1b7ea2121e8cfc))

- **42**: Capture phase context
  ([`5473b9b`](https://github.com/tlancaster6/AquaPose/commit/5473b9bda9073b8abeb67cd9798fd1d4dbddbb44))

- **42**: Create phase plan
  ([`35638d7`](https://github.com/tlancaster6/AquaPose/commit/35638d75e2329c0bff40ea8dc0baf2f9a8642de1))

- **42-01**: Complete baseline measurement plan
  ([`5054377`](https://github.com/tlancaster6/AquaPose/commit/50543775fdf7c5293aad4f13009d8ac05d31e4eb))

- **43**: Capture phase context
  ([`092a358`](https://github.com/tlancaster6/AquaPose/commit/092a3587b7f07a3e408bc9a990345c8537ded302))

- **43-01**: Complete helper extraction plan - utils.py created
  ([`016038e`](https://github.com/tlancaster6/AquaPose/commit/016038e11782de09864932f4f2af1ed0d28fe38c))

- **43-02**: Complete DltBackend plan
  ([`5cdab52`](https://github.com/tlancaster6/AquaPose/commit/5cdab52cf91ffb0e9783a83b682d7ce83d06d0a1))

- **43.1**: Capture phase context
  ([`fb286b3`](https://github.com/tlancaster6/AquaPose/commit/fb286b3b43d70b3adb1665f7dc4305c98bc8ad0b))

- **43.1**: Complete phase research
  ([`fc96905`](https://github.com/tlancaster6/AquaPose/commit/fc96905f3f8fef4e3427ec0369b2e84b9d02a1b4))

- **43.1**: Create phase plans and update requirements
  ([`45c44ff`](https://github.com/tlancaster6/AquaPose/commit/45c44fffeb1c7c050342642ec7bf2df461cd7856))

- **43.1**: Research phase domain
  ([`b1b8c70`](https://github.com/tlancaster6/AquaPose/commit/b1b8c7046278e8280229d0a730ad39a8a1823af4))

- **43.1-01**: Complete plan execution summary
  ([`01ffe7c`](https://github.com/tlancaster6/AquaPose/commit/01ffe7c5739c5236f9da4d24500b0db6b6ffec13))

- **43.1-02**: Complete plan execution summary
  ([`f307f05`](https://github.com/tlancaster6/AquaPose/commit/f307f05e3dbf5ac0cb47ca91ea8f0303174bc38d))

- **44**: Capture phase context
  ([`b7f9857`](https://github.com/tlancaster6/AquaPose/commit/b7f985787e0f9e9a5ef2151ca471ed69310edff4))

- **44-01**: Complete plan execution summary and update tracking
  ([`bce98c2`](https://github.com/tlancaster6/AquaPose/commit/bce98c239a605b7542518745d2d09546b9783b77))

- **45**: Capture phase context
  ([`fe6b278`](https://github.com/tlancaster6/AquaPose/commit/fe6b2788496cc10faca7d296a83d3a56e062dad7))

- **45**: Create phase plan
  ([`f4921dd`](https://github.com/tlancaster6/AquaPose/commit/f4921dd35311f4c86e59e83f398111db70a7a706))

- **45**: Research phase domain
  ([`b1f9b4a`](https://github.com/tlancaster6/AquaPose/commit/b1f9b4abdfa1711c70ba07227fa19dc4fd0c1fef))

- **45-01**: Add plan summary
  ([`6247179`](https://github.com/tlancaster6/AquaPose/commit/6247179d13701e9ea81d1e6ed4d33790b400c6d2))

- **45-02**: Add plan summary
  ([`1e8d4a3`](https://github.com/tlancaster6/AquaPose/commit/1e8d4a3ed700228eb7f9667f1c75ae04422e0565))

- **phase-40**: Complete phase execution
  ([`ac4c719`](https://github.com/tlancaster6/AquaPose/commit/ac4c71977c32b0444d133c79d2387616765a44cd))

- **phase-41**: Complete phase execution
  ([`5a10bba`](https://github.com/tlancaster6/AquaPose/commit/5a10bbad0c57dc36368726ad47cfd5971662cef0))

- **phase-42**: Complete phase execution
  ([`7445722`](https://github.com/tlancaster6/AquaPose/commit/744572222a0ec117d8c8263d5679ff5cb1e1cada))

- **phase-43**: Complete phase execution
  ([`6458689`](https://github.com/tlancaster6/AquaPose/commit/64586899c1403f41526325451dcb4520f3ad97ad))

- **phase-43.1**: Complete phase execution
  ([`658841e`](https://github.com/tlancaster6/AquaPose/commit/658841ebac0f64b0a4116f2bcc8011d7f36aac11))

- **quick-15**: Complete soft scoring kernel plan execution
  ([`df0e470`](https://github.com/tlancaster6/AquaPose/commit/df0e470d8e4e69f6135a5acff40d40f7588fd8ef))

- **quick-15**: Create plan for soft scoring kernel and ghost penalty removal
  ([`736fd6a`](https://github.com/tlancaster6/AquaPose/commit/736fd6a80adda6cf3922a53f25d5da4d3cf37412))

- **quick-15**: Replace binary inlier counting with soft linear scoring kernel and remove ghost
  penalty
  ([`3640a54`](https://github.com/tlancaster6/AquaPose/commit/3640a543fd32dfdc2746f6146d0eeaf59c7165e0))

- **quick-16**: Complete plan execution summary
  ([`f5a40a4`](https://github.com/tlancaster6/AquaPose/commit/f5a40a479f409224028b20bd4e5863f4e114f7d4))

- **quick-16**: Restructure tune_association.py with joint 2D grid sweep and widened ranges
  ([`ea852f0`](https://github.com/tlancaster6/AquaPose/commit/ea852f0071c06e56affecc53b023715fff6a7d6f))

- **quick-17**: Complete plan execution summary
  ([`473a86a`](https://github.com/tlancaster6/AquaPose/commit/473a86abe791622a7a29a146b1bb529f06074e11))

- **quick-17**: Create plan to unify n_sample_points config
  ([`5961266`](https://github.com/tlancaster6/AquaPose/commit/5961266f7c39579745bd0a1d79acbbe3e2ba2575))

- **state**: Mark phase 45 complete, v3.1 milestone shipped
  ([`f1dce77`](https://github.com/tlancaster6/AquaPose/commit/f1dce774276cc47dc5fcabc8dd62a597435da1b5))

- **state**: Record phase 43.1 context session
  ([`f12c840`](https://github.com/tlancaster6/AquaPose/commit/f12c8401b1d22f887f54032e208bda87e976fb4e))

- **state**: Record phase 45 context session
  ([`0aa20fa`](https://github.com/tlancaster6/AquaPose/commit/0aa20fa0adea4d86c41de75b65d37fe23847ec17))

### Features

- Add reconstruction coverage rate to evaluation metrics
  ([`044112d`](https://github.com/tlancaster6/AquaPose/commit/044112d01fefa1f7b7891f7be521529e301c00ef))

- **40-01**: Add MidlineSet assembly and NPZ export to DiagnosticObserver
  ([`e61e002`](https://github.com/tlancaster6/AquaPose/commit/e61e0025d3f5e88499916fd57384343f3b083ea5))

- **40-01**: Create MidlineFixture dataclass and NPZ key convention
  ([`10d7b5f`](https://github.com/tlancaster6/AquaPose/commit/10d7b5f44e951c125b61b0f3cde4f56764c91679))

- **40-02**: Implement load_midline_fixture for NPZ deserialization
  ([`7ec4f96`](https://github.com/tlancaster6/AquaPose/commit/7ec4f967d0667c80683ae90a60090754a9a4a283))

- **41-01**: Add CalibBundle dataclass and v2.0 NPZ loading support
  ([`e2aa926`](https://github.com/tlancaster6/AquaPose/commit/e2aa926ba4082a3c6b445d4be9f130d53991ae6b))

- **41-01**: Extend export_midline_fixtures to write calib/ keys for v2.0
  ([`e6b3369`](https://github.com/tlancaster6/AquaPose/commit/e6b33696adbec5394c4abfdc95cf1c2febea1f68))

- **41-02**: Implement format_summary_table and write_regression_json
  ([`855f593`](https://github.com/tlancaster6/AquaPose/commit/855f5931eb8d9bdedb03ca06be3f52b6af7c3b35))

- **41-02**: Implement run_evaluation harness orchestrator
  ([`57a32b1`](https://github.com/tlancaster6/AquaPose/commit/57a32b18d193af3c6fbcd06b9691f87569b475b1))

- **41-02**: Implement select_frames, compute_tier1, compute_tier2
  ([`a81993d`](https://github.com/tlancaster6/AquaPose/commit/a81993d26f870fcacb97463a1385c1937699e3de))

- **42-01**: Add flag_outliers and format_baseline_report to evaluation output
  ([`b0d7999`](https://github.com/tlancaster6/AquaPose/commit/b0d79993f97bf585bd9beba8be04dc022ed399a6))

- **42-01**: Create measure_baseline.py baseline measurement script
  ([`fb1fe2f`](https://github.com/tlancaster6/AquaPose/commit/fb1fe2f136f1ad62cb7901a14aa6eda226bb2a3d))

- **43-01**: Extract shared reconstruction helpers to utils.py
  ([`4777375`](https://github.com/tlancaster6/AquaPose/commit/47773756548ec530759755f3ae0f1bceeb9a7c72))

- **43-02**: Implement DltBackend with confidence-weighted DLT triangulation
  ([`210c0e7`](https://github.com/tlancaster6/AquaPose/commit/210c0e781b79042442f9394728bf3444173c53e5))

- **43-02**: Register DltBackend in backends/__init__.py factory
  ([`1c54c11`](https://github.com/tlancaster6/AquaPose/commit/1c54c11e2d6a8f48a0d68de7030fde13e6fb5c40))

- **43.1-01**: Add skip_tier2 flag and generate_fixture helper to evaluation harness
  ([`682b3f0`](https://github.com/tlancaster6/AquaPose/commit/682b3f0c997860a564a9369aa28b6ad97bd8a831))

- **43.1-02**: Create association parameter tuning sweep script
  ([`e1b3fc8`](https://github.com/tlancaster6/AquaPose/commit/e1b3fc80984861c03ea99d5d37115c72e0540d63))

- **44-01**: Add outlier_threshold to ReconstructionConfig and wire through run_evaluation
  ([`83631bc`](https://github.com/tlancaster6/AquaPose/commit/83631bc583d49e3a2ebb6481422da79a736b04e2))

- **44-01**: Create grid search and comparison script for DLT threshold tuning
  ([`ca560f8`](https://github.com/tlancaster6/AquaPose/commit/ca560f8790fdd522863ced477c0841382ebaab9b))

- **45-01**: Delete dead reconstruction backends and update production code
  ([`8f9983b`](https://github.com/tlancaster6/AquaPose/commit/8f9983b1728c5e3cb11455c607672182b36e3088))

- **45-02**: Update tests and docs for DLT-only reconstruction
  ([`42c72e7`](https://github.com/tlancaster6/AquaPose/commit/42c72e78fd6167ffccd63a6ad411dae99bd63dfb))

- **quick-15**: Replace binary inlier counting with soft kernel, remove ghost penalty
  ([`f76e676`](https://github.com/tlancaster6/AquaPose/commit/f76e6768bdd5ef4456f487b21fadd2fdb344dbfd))

- **quick-16**: Restructure tune_association.py with joint grid sweep
  ([`4905671`](https://github.com/tlancaster6/AquaPose/commit/4905671f4e4365f0695835caae5975a7be4ea65c))

- **quick-17**: Remove N_SAMPLE_POINTS constant and update all consumers
  ([`07a8314`](https://github.com/tlancaster6/AquaPose/commit/07a8314432d759588dc2b452c57c43be2d5a163f))

- **quick-17**: Update config hierarchy for n_sample_points
  ([`4cc3f81`](https://github.com/tlancaster6/AquaPose/commit/4cc3f81b4e8f678cc94c0a66c92c33ac3a8aa61c))

### Testing

- **40-02**: Add failing tests for load_midline_fixture round-trip and validation
  ([`ce66995`](https://github.com/tlancaster6/AquaPose/commit/ce669954babf8abd4531be02095d8190165d4fe5))

- **41-01**: Add failing tests for CalibBundle and v2.0 NPZ loading
  ([`2c99a75`](https://github.com/tlancaster6/AquaPose/commit/2c99a75f6f5a1c78586a2db359b58dae8a5d0d3a))

- **41-01**: Add failing tests for export_midline_fixtures v2.0 with models
  ([`4a90ff7`](https://github.com/tlancaster6/AquaPose/commit/4a90ff7006cdec3211e8c7320cd5a4b343d7cfb2))

- **41-02**: Add failing tests for format_summary_table and write_regression_json
  ([`5882338`](https://github.com/tlancaster6/AquaPose/commit/58823382e576c525045c8dc322b283c3efabfc20))

- **41-02**: Add failing tests for run_evaluation harness
  ([`9fddea0`](https://github.com/tlancaster6/AquaPose/commit/9fddea091fbd21d1147a2561b5e1353f4183f928))

- **41-02**: Add failing tests for select_frames and metric computation
  ([`491ace4`](https://github.com/tlancaster6/AquaPose/commit/491ace42a7e2e0fd0729842f98fd5e1680f1020a))

- **quick-15**: Update scoring tests for soft kernel, remove ghost penalty tests
  ([`d9e48a7`](https://github.com/tlancaster6/AquaPose/commit/d9e48a72784bd1c924980def728ff0488b92f497))


## v1.1.0-dev.5 (2026-03-02)

### Bug Fixes

- Use python3 in import boundary hook and fix pipeline config test paths
  ([`c2df444`](https://github.com/tlancaster6/AquaPose/commit/c2df444384963d441e5434b11bf27495440d31b9))

### Chores

- Cleanup and planning for next milestone
  ([`b9fe845`](https://github.com/tlancaster6/AquaPose/commit/b9fe84558a7eff50e3a9eb05839902d815bbf4f6))

- Complete v3.0 Ultralytics Unification milestone
  ([`065c59e`](https://github.com/tlancaster6/AquaPose/commit/065c59e1b45787bbf5bba6a398d5e3a9ddcf14db))

- Removing legacy scripts
  ([`4754ce4`](https://github.com/tlancaster6/AquaPose/commit/4754ce45d9db3bb44c222962646a480fa5878105))

- **38-04**: Delete visualization/diagnostics.py dead code
  ([`846b157`](https://github.com/tlancaster6/AquaPose/commit/846b1574f142f75d9dead2825e72b77bf00b2179))

### Documentation

- **38**: Add phase research and context
  ([`86f3665`](https://github.com/tlancaster6/AquaPose/commit/86f3665898fd0b0a21e9b0882e4188dfc2bc35d6))

- **38**: Capture phase context
  ([`1d6b0d4`](https://github.com/tlancaster6/AquaPose/commit/1d6b0d4bc630953017c270904921ee2c1f024804))

- **38**: Create phase plan
  ([`ea098a4`](https://github.com/tlancaster6/AquaPose/commit/ea098a4780c487c2e52908035ee58f4fdae25da4))

- **38**: Research phase domain
  ([`66cfcee`](https://github.com/tlancaster6/AquaPose/commit/66cfceecc88904e98a93c49527b2416c77c45e21))

- **38-01**: Complete config field consolidation plan
  ([`88ee4b8`](https://github.com/tlancaster6/AquaPose/commit/88ee4b83660829760cc69703e2e2efc3c07d3286))

- **38-02**: Complete NDJSON-to-txt+yaml migration plan
  ([`eecf857`](https://github.com/tlancaster6/AquaPose/commit/eecf857aae79dd93275b9f3f5cc6d93b5f015d70))

- **38-04**: Complete dead code analysis and cleanup plan
  ([`97e87e5`](https://github.com/tlancaster6/AquaPose/commit/97e87e5df7853d63ad12bf9b4eb2dcab149730bf))

- **39**: Capture phase context
  ([`0f7b317`](https://github.com/tlancaster6/AquaPose/commit/0f7b3173a3324a227e42d6a84da227a85f0ace90))

- **39**: Create phase plan
  ([`52e7cd1`](https://github.com/tlancaster6/AquaPose/commit/52e7cd12a8182b47d6dac3b7693ac323a1f06dc9))

- **39**: Research phase domain
  ([`b8403c2`](https://github.com/tlancaster6/AquaPose/commit/b8403c225fd36d0d4c6fdf4584d33956944e1418))

- **39-01**: Complete core/types package and implementation file relocation plan
  ([`0264e06`](https://github.com/tlancaster6/AquaPose/commit/0264e06bf484d24ccb901b0b9f3c87de799fbd66))

- **39-02**: Complete import migration plan execution
  ([`a720686`](https://github.com/tlancaster6/AquaPose/commit/a720686b4c139c496af86791b12e10a7ac2bca7f))

- **39-03**: Complete test import migration plan
  ([`e01082c`](https://github.com/tlancaster6/AquaPose/commit/e01082c59468f0480e1b8f28a884df26de80e73e))

- **39-04**: Complete documentation and docstring cleanup plan
  ([`18097c2`](https://github.com/tlancaster6/AquaPose/commit/18097c252907dfa3f14de1e22909ad618032e795))

- **39-04**: Remove stale legacy path references from module docstrings
  ([`a73c66a`](https://github.com/tlancaster6/AquaPose/commit/a73c66aea4f202f36985747bb30611b27618721c))

- **39-04**: Update GUIDEBOOK.md and CLAUDE.md with post-migration structure
  ([`db53f44`](https://github.com/tlancaster6/AquaPose/commit/db53f4423a6e0272d30f15457b91b16684ba1ace))

- **phase-38**: Complete phase execution
  ([`44a5d62`](https://github.com/tlancaster6/AquaPose/commit/44a5d620af5053ba5a9de090fa2373482d27a2fd))

- **phase-39**: Complete phase execution
  ([`0fcc5eb`](https://github.com/tlancaster6/AquaPose/commit/0fcc5eb7fe0cd7417f5a303d1bbf1e7a49632c11))

- **state**: Record phase 38 context session
  ([`e9bb7e6`](https://github.com/tlancaster6/AquaPose/commit/e9bb7e67acf5f33512951af4d1f2826254ce8e80))

- **v3.0**: Complete milestone audit — 16/16 requirements satisfied
  ([`83cd535`](https://github.com/tlancaster6/AquaPose/commit/83cd5359145a04f36fa66b68eb3ad5659feb4c35))

### Features

- **38-01**: Consolidate config fields — model_path -> weights_path, remove keypoint_weights_path
  ([`f833d04`](https://github.com/tlancaster6/AquaPose/commit/f833d04ac52dabfe728e92cf7f186c9730fa14ea))

- **38-01**: Fix init-config defaults to reflect Ultralytics-based architecture
  ([`69bd1db`](https://github.com/tlancaster6/AquaPose/commit/69bd1dbb6c3ea516e751ec921f1fcf90964271d1))

- **38-02**: Migrate build script to YOLO txt+yaml output format
  ([`69bfe02`](https://github.com/tlancaster6/AquaPose/commit/69bfe02901afde341f2289ad67d5f3a011f961f4))

- **38-04**: Produce import analysis dead code report for legacy directories
  ([`d36d1b2`](https://github.com/tlancaster6/AquaPose/commit/d36d1b2e3483ef114560d1801bcc0e692c3b5421))

- **39-01**: Create core/types/ package with cross-stage shared types
  ([`6dea193`](https://github.com/tlancaster6/AquaPose/commit/6dea193e408b9aaf179a0582ebd589e5cc157fa9))

- **39-01**: Merge YOLODetector and make_detector() into detection backend
  ([`80af32f`](https://github.com/tlancaster6/AquaPose/commit/80af32f7ee940d42660d6d401d975bb45f11119a))

- **39-01**: Relocate implementation files to new core locations
  ([`26bae8f`](https://github.com/tlancaster6/AquaPose/commit/26bae8ffa95c0ca35806c8c0548383a1e114f7a9))

- **39-02**: Delete legacy reconstruction/, segmentation/, tracking/ directories
  ([`9236537`](https://github.com/tlancaster6/AquaPose/commit/92365374431b94b4b2b5f897545120619b2bb550))

- **39-02**: Rewire all src/ imports to core paths; delete shim files
  ([`ed0fcd9`](https://github.com/tlancaster6/AquaPose/commit/ed0fcd94763c23436864e5027438cc3b14994912))

- **39-03**: Update all test file imports to new core paths
  ([`ba52398`](https://github.com/tlancaster6/AquaPose/commit/ba523983f471045e700eadd6320741b700909c9b))


## v1.1.0-dev.4 (2026-03-02)

### Bug Fixes

- Debugging image-space to crop-space coordinate transforms
  ([`f106a0d`](https://github.com/tlancaster6/AquaPose/commit/f106a0dde99fd306d8d7ae1f7f19ceeb8d87faf3))

- Synthetic frame fallback, video-free overlays, and v2.1 cleanup
  ([`439d7be`](https://github.com/tlancaster6/AquaPose/commit/439d7bed768b5382033ac8c42b0fc546ad19cbb9))

- Wire DiagnosticObserver to auto-export centroid correspondences on pipeline complete
  ([`25ba258`](https://github.com/tlancaster6/AquaPose/commit/25ba258e57253183547fc9dbf611df19abe1b976))

- **30**: Default output_dir to project_dir/runs/<run_id> when project_dir is set
  ([`cc7a23d`](https://github.com/tlancaster6/AquaPose/commit/cc7a23d866357366bdca6e1e28cc0158992e63e1))

- **30**: Update observer_factory to use config.stop_frame instead of config.detection.stop_frame
  ([`0a6d2a3`](https://github.com/tlancaster6/AquaPose/commit/0a6d2a3d22a4aa1b7f36737ca0a57c1c399eaef9))

- **overlay**: Correct bbox (x,y,w,h) handling and remove dead _draw_bbox method
  ([`ab5f9b0`](https://github.com/tlancaster6/AquaPose/commit/ab5f9b0fef917e1a8512d9f014526ab3b38f01bb))

- **overlay**: Enable bbox drawing in diagnostic mode and unwrap AnnotatedDetection
  ([`415b04c`](https://github.com/tlancaster6/AquaPose/commit/415b04c81891f88844341a42fde05a78940dafbb))

- **pose**: Mask background in affine crops to match training letterbox
  ([`e316911`](https://github.com/tlancaster6/AquaPose/commit/e31691187fd9692093ef47ab59b90982f0fe6e2e))

- **pose**: Match training/inference crop geometry for DirectPoseBackend
  ([`d4391bd`](https://github.com/tlancaster6/AquaPose/commit/d4391bdb0305976ce3339d079918ff6848bd2382))

- **quick-12**: Fix type error in affine_warp_crop for cv2.getAffineTransform
  ([`193feb0`](https://github.com/tlancaster6/AquaPose/commit/193feb02b2758b8b81ea50c6867b54f321ffb986))

- **quick-13**: Change OBB NDJSON key from "obbs" to "annotations"
  ([`93d2b7f`](https://github.com/tlancaster6/AquaPose/commit/93d2b7f4cdb51e92659a23a6a80d2d50149f8313))

### Chores

- Archive v2.2 phase directories (6 phases)
  ([`5eadd39`](https://github.com/tlancaster6/AquaPose/commit/5eadd3934a954bb03eef7eaba9832e1108bbcd57))

- **training**: Update default OBB model from yolov8s-obb to yolo26n-obb
  ([`9fa8684`](https://github.com/tlancaster6/AquaPose/commit/9fa86844b5e1bb87bedb567688dfeb59f136ada9))

### Documentation

- Capture todo - Add plain YOLO training path to CLI
  ([`cd6a062`](https://github.com/tlancaster6/AquaPose/commit/cd6a062e9631718cdbc6ffc82795c29320ed3aea))

- Capture todo - consolidate weights_path and keypoint_weights_path config fields
  ([`046aed0`](https://github.com/tlancaster6/AquaPose/commit/046aed03e215772314d20f5a1270a24b0787e772))

- Capture todo - Replace custom U-Net with Ultralytics-only model stack
  ([`9c7d50b`](https://github.com/tlancaster6/AquaPose/commit/9c7d50b8a07db0dadb6e4de32ec485c98453e38b))

- Capture todo - Standardize model naming conventions in CLI and repo
  ([`36db7cb`](https://github.com/tlancaster6/AquaPose/commit/36db7cbcf37570070bd1c22036669e104905d60b))

- Complete v2.2 Backends research
  ([`fcbbba2`](https://github.com/tlancaster6/AquaPose/commit/fcbbba2f4e0e7ebac00e2ac70e69eef827fb030a))

- Complete v3.0 Ultralytics Unification research
  ([`1b2f3db`](https://github.com/tlancaster6/AquaPose/commit/1b2f3dba560d38f363fa56dd1d3c220d7b55010b))

- Create milestone v2.2 roadmap (6 phases)
  ([`c8be407`](https://github.com/tlancaster6/AquaPose/commit/c8be4071b29f026c1c63c1a46e9d141d708b69ee))

- Create milestone v3.0 roadmap (3 phases)
  ([`f70c560`](https://github.com/tlancaster6/AquaPose/commit/f70c56067f49fa0b4d2165d9e5d5bde15e11793d))

- Define milestone v2.2 requirements
  ([`162c886`](https://github.com/tlancaster6/AquaPose/commit/162c88667d415320181a5858f49fa204eab4d0b4))

- Define milestone v3.0 requirements (12 requirements)
  ([`35c7303`](https://github.com/tlancaster6/AquaPose/commit/35c730321c4ea27133290748656466ce2e59cc2a))

- Fix roadmap plan listings and progress table for phases 29-34
  ([`8a97b3d`](https://github.com/tlancaster6/AquaPose/commit/8a97b3d042b0dbe3a960bcc7352e738d7d017b45))

- Start milestone v2.2 Backends
  ([`5403629`](https://github.com/tlancaster6/AquaPose/commit/540362969960453e9d2859dca57302e079d9a632))

- Start milestone v3.0 Ultralytics Unification
  ([`d9c9cb2`](https://github.com/tlancaster6/AquaPose/commit/d9c9cb21b490630b6e7cf0e65f215375f73ba031))

- **10**: Plan for storing 3D consensus centroids on TrackletGroup
  ([`ae71423`](https://github.com/tlancaster6/AquaPose/commit/ae71423ba23c08a1543dcd57295500ea2d80377c))

- **10-01**: Complete store-3d-consensus-centroids-on-tracklet quick task
  ([`ab90ce2`](https://github.com/tlancaster6/AquaPose/commit/ab90ce2aed02e63672469e98437d56c272f36440))

- **11**: Complete quick task 11 — training visualization grids
  ([`5bf3fc4`](https://github.com/tlancaster6/AquaPose/commit/5bf3fc48aae07d541fa077da6a67fa0290385e52))

- **29**: Capture phase context
  ([`6f94652`](https://github.com/tlancaster6/AquaPose/commit/6f946526e858417b928064c4505dd2df97b7e1ab))

- **29-01**: Audit and fix stale content in GUIDEBOOK.md
  ([`a6e2573`](https://github.com/tlancaster6/AquaPose/commit/a6e257363a9bb87c5265fc6b1aa242c8ddefa989))

- **29-01**: Complete Audit and Fix Stale Content plan
  ([`78b00e7`](https://github.com/tlancaster6/AquaPose/commit/78b00e72417625399b9e657cff629a20fd2e35e0))

- **29-02**: Add backend registry subsection to Section 8
  ([`31be607`](https://github.com/tlancaster6/AquaPose/commit/31be607f62b2422a73e1999bfeb5816076f2e6bf))

- **29-02**: Add v2.2 planned feature inline tags throughout GUIDEBOOK
  ([`236fd4b`](https://github.com/tlancaster6/AquaPose/commit/236fd4b9567e8b9cbead992d95ca8f3bc47d6416))

- **29-02**: Complete Add v2.2 Planned Features plan
  ([`8418d57`](https://github.com/tlancaster6/AquaPose/commit/8418d570811bb2dabea43f8ce2c9d6ddddc7d92a))

- **30**: Capture phase context
  ([`a5eb66f`](https://github.com/tlancaster6/AquaPose/commit/a5eb66fa35de249322b81ccdacf96fe21b4b7252))

- **30**: Complete phase research, plans, and verification
  ([`9271f5d`](https://github.com/tlancaster6/AquaPose/commit/9271f5dddd56af96ed2926e8ccb6d658740b5642))

- **30**: Research phase domain
  ([`c46d748`](https://github.com/tlancaster6/AquaPose/commit/c46d74848eda11f2940216eadf2d7041d72dfb77))

- **30-01**: Complete extend dataclasses and universalize config validation plan
  ([`eeef9c7`](https://github.com/tlancaster6/AquaPose/commit/eeef9c7f5aa915a893894aeed73c11e668f346f4))

- **30-02**: Complete Config Promotion and Propagation plan
  ([`f7a29d3`](https://github.com/tlancaster6/AquaPose/commit/f7a29d3d56aea27390345eb1e7e8a8fd47f1068b))

- **30-03**: Complete CLI Project Scaffold and Path Resolution plan
  ([`f8ac74d`](https://github.com/tlancaster6/AquaPose/commit/f8ac74d1de674c717a10ce71874648cd374b5c15))

- **31**: Capture phase context
  ([`1482889`](https://github.com/tlancaster6/AquaPose/commit/1482889695c5b5330717be9a222f6a264260c683))

- **31**: Create phase plan for training infrastructure
  ([`4afc73e`](https://github.com/tlancaster6/AquaPose/commit/4afc73efffe427b8c9914286927b3d67ff49e6c0))

- **31-01**: Complete training infrastructure foundation plan
  ([`fa6066d`](https://github.com/tlancaster6/AquaPose/commit/fa6066dcf638d6f596717ad8f42357575c200dcb))

- **31-02**: Complete YOLO-OBB and pose training subcommands plan
  ([`7455db6`](https://github.com/tlancaster6/AquaPose/commit/7455db623c517cc17de4ce80f25e9f4c3a1be846))

- **32**: Capture phase context
  ([`00fd40a`](https://github.com/tlancaster6/AquaPose/commit/00fd40a71b3df7af59afcef7d813c4ca4278ed91))

- **32**: Research YOLO-OBB detection backend phase
  ([`d76e523`](https://github.com/tlancaster6/AquaPose/commit/d76e52391afbf1b2a531362f1f4b72cfaed3c043))

- **32-01**: Complete YOLO-OBB backend and affine crop utilities plan
  ([`f2fd60e`](https://github.com/tlancaster6/AquaPose/commit/f2fd60e9e87ac959773b5f70c3617a515468b863))

- **32-02**: Complete OBB visualization extensions plan
  ([`0712f66`](https://github.com/tlancaster6/AquaPose/commit/0712f6655b461ce1499654f8a2b58d94877eec65))

- **33**: Capture phase context
  ([`b5c6e48`](https://github.com/tlancaster6/AquaPose/commit/b5c6e48c01b987e2e452232d61a6e5b85a73887c))

- **33**: Research and plan keypoint midline backend phase
  ([`ead451d`](https://github.com/tlancaster6/AquaPose/commit/ead451d93a8b2912a533c1094267ebec80c128d6))

- **33**: Research keypoint midline backend phase
  ([`67c7b8e`](https://github.com/tlancaster6/AquaPose/commit/67c7b8e915eb4a5de934109222e5ef61ff977f00))

- **33-01**: Complete DirectPoseBackend plan - SUMMARY, STATE, ROADMAP, REQUIREMENTS
  ([`48f4a29`](https://github.com/tlancaster6/AquaPose/commit/48f4a29c48b8f80cbab0bf5dc5be9c318c8404f3))

- **33-02**: Complete confidence-weighted reconstruction plan
  ([`4b1e75d`](https://github.com/tlancaster6/AquaPose/commit/4b1e75d26098b21fb3aca9002d2a5d921af774dc))

- **33.1**: Capture phase context
  ([`449e098`](https://github.com/tlancaster6/AquaPose/commit/449e0987cc1b11f68411202d6720ebde574cffab))

- **33.1**: Create phase plan for keypoint training data augmentation
  ([`5cc8fdd`](https://github.com/tlancaster6/AquaPose/commit/5cc8fddd34744e1bf3e1c90a847beb63871d7109))

- **33.1**: Research phase domain
  ([`c7681a9`](https://github.com/tlancaster6/AquaPose/commit/c7681a921e0cab2131afe7df2ea8fe76db090115))

- **33.1-01**: Complete keypoint augmentation plan - SUMMARY, STATE, ROADMAP
  ([`c342616`](https://github.com/tlancaster6/AquaPose/commit/c342616a2a04a353ee4ab9ed7bab60ff323ba831))

- **35**: Capture phase context
  ([`c200477`](https://github.com/tlancaster6/AquaPose/commit/c200477347482fe3c5f3419d314e33536c02f143))

- **35**: Create phase plan
  ([`871d47a`](https://github.com/tlancaster6/AquaPose/commit/871d47a8a641724d341a5eab819febc93516bff4))

- **35**: Research phase domain
  ([`6ba6292`](https://github.com/tlancaster6/AquaPose/commit/6ba6292a420d0c5b1252410631261a3bf558cb9e))

- **35-02**: Complete plan — midline backend stubs, CLEAN-03 done, Phase 35 complete
  ([`a18212d`](https://github.com/tlancaster6/AquaPose/commit/a18212d5de0a91ddfb020025024d12738913ac03))

- **36**: Capture phase context
  ([`dd7a169`](https://github.com/tlancaster6/AquaPose/commit/dd7a1690407aae2a9b3dc5f4d0f42ef127b94995))

- **36**: Create phase plan
  ([`e640a13`](https://github.com/tlancaster6/AquaPose/commit/e640a134d4e92f1cc38b6343cee310c54ec79f54))

- **36**: Research phase domain
  ([`e56efe9`](https://github.com/tlancaster6/AquaPose/commit/e56efe9b463b4546a5f6b876cd618d6cd7a382e3))

- **36-01**: Complete seg data converter plan execution
  ([`f6bb4b5`](https://github.com/tlancaster6/AquaPose/commit/f6bb4b5dbb87835edd7023fcbe1ad6d679c0297b))

- **36-02**: Complete training wrappers plan execution
  ([`354831b`](https://github.com/tlancaster6/AquaPose/commit/354831b8e916662c0a6f3404af893d7580d4ffb1))

- **37**: Capture phase context
  ([`b41b482`](https://github.com/tlancaster6/AquaPose/commit/b41b482496c02c652572586c5fa36bac62bdf40c))

- **37**: Create phase plan for pipeline integration
  ([`f81bbdb`](https://github.com/tlancaster6/AquaPose/commit/f81bbdb4cf5108ddc9a9f7996723c7b7894ec2d6))

- **37**: Research pipeline integration phase
  ([`face555`](https://github.com/tlancaster6/AquaPose/commit/face555cdc8b7130834a55e51a42124dc2f0836f))

- **37**: Update stale docstrings — backends are no longer stubs
  ([`8b16979`](https://github.com/tlancaster6/AquaPose/commit/8b169798fdc6a7d8a4ee9d3e2d838fc2db0988d4))

- **37-01**: Complete pipeline-integration plan 01 - backend renaming
  ([`c61fb4f`](https://github.com/tlancaster6/AquaPose/commit/c61fb4f5c70b0b4e3bebeda53e6243e972ef82f0))

- **37-02**: Complete pipeline-integration plan 02 - YOLO backend implementation
  ([`44cce73`](https://github.com/tlancaster6/AquaPose/commit/44cce730753e6b11b78f15bb356ecbb3759019c8))

- **phase-29**: Complete phase verification and finalize
  ([`bcc0710`](https://github.com/tlancaster6/AquaPose/commit/bcc0710ca078772ae359d9746bf4d94393d5661f))

- **phase-30**: Complete phase execution and verification
  ([`e782201`](https://github.com/tlancaster6/AquaPose/commit/e78220152be4d789d0129defe9e61126b68c4441))

- **phase-31**: Complete phase execution, resolve doc gaps, mark verified
  ([`14c7e02`](https://github.com/tlancaster6/AquaPose/commit/14c7e02ab1c9ec87d3c04f1c2640f22f821143dd))

- **phase-32**: Complete phase execution, mark verified
  ([`7b79f3e`](https://github.com/tlancaster6/AquaPose/commit/7b79f3e26fe3d18f5acc4349e89af682b7e810e5))

- **phase-33**: Complete phase execution, mark verified
  ([`53a9ebb`](https://github.com/tlancaster6/AquaPose/commit/53a9ebb836b2b9370282fc0fc94828cc6e31c6d8))

- **phase-33.1**: Complete phase execution — augmentation requirements and traceability
  ([`c3046bd`](https://github.com/tlancaster6/AquaPose/commit/c3046bd97524a446cf773487573b5eee63092e1e))

- **phase-35**: Complete phase execution
  ([`00e477f`](https://github.com/tlancaster6/AquaPose/commit/00e477f3934607d1502e375e8131361b831f4542))

- **phase-36**: Complete phase execution
  ([`56b6f00`](https://github.com/tlancaster6/AquaPose/commit/56b6f0095bb4b0279a3dfee6faab66ad818ccea8))

- **phase-37**: Complete phase execution
  ([`c6a4b50`](https://github.com/tlancaster6/AquaPose/commit/c6a4b50b7b1de975c3ec0ccdcfaea5743b4b6f1f))

- **quick-10**: Store 3D consensus centroids on TrackletGroup
  ([`64aef3b`](https://github.com/tlancaster6/AquaPose/commit/64aef3b9343bf17e3eff0ba4e207e9e7deb716d3))

- **quick-11**: Training visualization grids for pose and unet
  ([`bd50e37`](https://github.com/tlancaster6/AquaPose/commit/bd50e37249f2ea41576ba4f189fb55895a208ec6))

- **quick-12**: Complete YOLO training data builder plan
  ([`8cc5508`](https://github.com/tlancaster6/AquaPose/commit/8cc55083fcedc7de000ccf65087406292b33fcef))

- **quick-12**: YOLO-OBB and YOLO-pose training set generation
  ([`4854d0c`](https://github.com/tlancaster6/AquaPose/commit/4854d0c3da25f037dfd2a16edbf5e7fa85388ed4))

- **quick-13**: Complete yolo_obb NDJSON unification plan
  ([`901e83c`](https://github.com/tlancaster6/AquaPose/commit/901e83c022f01746ecc2b83353acf86884424479))

- **quick-13**: Convert yolo_obb.py to ndjson format and simplify
  ([`7810997`](https://github.com/tlancaster6/AquaPose/commit/781099787b48449642a543dacea95aa2acde3032))

- **quick-14**: Adopt Ultralytics-native NDJSON format
  ([`af5f552`](https://github.com/tlancaster6/AquaPose/commit/af5f552ca77cd7e3d9ed1f30b11643880b553ac9))

- **quick-14**: Complete adopt-ultralytics-native-ndjson-format-u task
  ([`72c1d2e`](https://github.com/tlancaster6/AquaPose/commit/72c1d2e67053d15a3a9f062bc6b702fa7cf4b3c8))

- **state**: Record phase 33.1 context session
  ([`f4043ad`](https://github.com/tlancaster6/AquaPose/commit/f4043adeca5db3a350a8ff680f461dc7551d9b0a))

### Features

- Support non-square 128x64 input for pose and U-Net training/inference
  ([`bd5ebb7`](https://github.com/tlancaster6/AquaPose/commit/bd5ebb7eaedf48a2aaedd7c9535658c63f43cfa2))

- **10-01**: Add consensus_centroids field to TrackletGroup and populate in refinement
  ([`b7a9922`](https://github.com/tlancaster6/AquaPose/commit/b7a99223531485dd97d233ecb0284a6ba677ef01))

- **10-01**: Add export_centroid_correspondences to DiagnosticObserver
  ([`e8124d1`](https://github.com/tlancaster6/AquaPose/commit/e8124d15811f253741ccf04a181126f6425bc8dc))

- **11-01**: Add training visualization grid utilities
  ([`cd1e191`](https://github.com/tlancaster6/AquaPose/commit/cd1e191554fdd3e7a8dfff24d5c9c59ca6cb8411))

- **11-02**: Wire viz calls into train_unet and train_pose
  ([`2881361`](https://github.com/tlancaster6/AquaPose/commit/288136187436e0fe0836fd30588598faa175ecc2))

- **30-01**: Extend Detection and Midline2D dataclasses with v2.2 fields
  ([`4e9a086`](https://github.com/tlancaster6/AquaPose/commit/4e9a086d3aa44a5ed9a361b7900670a1c508e70e))

- **30-01**: Universalize _filter_fields() with strict reject and rename hints
  ([`b3d4d5e`](https://github.com/tlancaster6/AquaPose/commit/b3d4d5e282a4681285ae8d312e46c9993e6f1bd9))

- **30-02**: Promote device, n_sample_points, stop_frame, project_dir to PipelineConfig
  ([`8bcb463`](https://github.com/tlancaster6/AquaPose/commit/8bcb463ccfea3c0410abbdf3c668b627fbe16096))

- **30-02**: Propagate device and n_sample_points through build_stages() and downstream modules
  ([`6f827b0`](https://github.com/tlancaster6/AquaPose/commit/6f827b03ecec354080abffbbb0e08e6f260a216f))

- **30-03**: Implement project_dir path resolution in load_config()
  ([`6411329`](https://github.com/tlancaster6/AquaPose/commit/6411329c46133acf248b50c41a01f443f86ea2e0))

- **30-03**: Rewrite init-config CLI with project scaffolding
  ([`b9fe86c`](https://github.com/tlancaster6/AquaPose/commit/b9fe86c2f339295e71634c22d8dcd07f3e572ec9))

- **31-01**: Create training/ package scaffold with common utilities and datasets
  ([`54fb2ad`](https://github.com/tlancaster6/AquaPose/commit/54fb2ad56f447ee977a71bf5909f3cceb169b2a8))

- **31-01**: Implement U-Net training subcommand with tests
  ([`a73b3f5`](https://github.com/tlancaster6/AquaPose/commit/a73b3f55f6c5cd38b99ab47660f2b2103e0b3df4))

- **31-02**: Add YOLO-OBB and pose training subcommands
  ([`45dadcf`](https://github.com/tlancaster6/AquaPose/commit/45dadcfa0e22ed840a67a0a8879d912b17067563))

- **31-02**: Migration cleanup — delete superseded segmentation files and update imports
  ([`9ec15bb`](https://github.com/tlancaster6/AquaPose/commit/9ec15bbb5849bd24869ac0128c247ce6672c2581))

- **32-01**: Add YOLOOBBBackend and extend detection backend registry
  ([`2a34192`](https://github.com/tlancaster6/AquaPose/commit/2a34192fea389120a06c8a0594400ed11bf7a09e))

- **32-01**: Implement affine crop utilities with invertible transform
  ([`71efa75`](https://github.com/tlancaster6/AquaPose/commit/71efa7551ee9be43daccf0a6f082539af02da78c))

- **32-02**: Extend Overlay2DObserver with OBB polygon rendering
  ([`0bf97d5`](https://github.com/tlancaster6/AquaPose/commit/0bf97d5cd64b90da8ba78d844be911a7f11dd82c))

- **32-02**: Extend TrackletTrailObserver with OBB polygon at trail head
  ([`760cf52`](https://github.com/tlancaster6/AquaPose/commit/760cf52a5a1ff3fb535690e22fcbdd5e9b669942))

- **33-01**: Extend MidlineConfig, wire build_stages, add prep CLI
  ([`0cc522d`](https://github.com/tlancaster6/AquaPose/commit/0cc522da2512f94914fdfc53165ff607a657c1db))

- **33-01**: Implement DirectPoseBackend with keypoint inference and NaN-padding
  ([`516fbf6`](https://github.com/tlancaster6/AquaPose/commit/516fbf6b989608e4026cba6763753bd81d61e248))

- **33-02**: Add confidence-weighted chamfer to curve_optimizer.py
  ([`12e274d`](https://github.com/tlancaster6/AquaPose/commit/12e274dc281e78db4fe49f0d762fde74373b289f))

- **33-02**: Add confidence-weighted DLT triangulation to triangulation.py
  ([`ae88279`](https://github.com/tlancaster6/AquaPose/commit/ae882791df26931350ed588269fff590cc1aa530))

- **33.1-01**: Add augmentation to KeypointDataset and masked loss utilities
  ([`f2c70ad`](https://github.com/tlancaster6/AquaPose/commit/f2c70ad0b067ace313c2a0b058ad956341ad7159))

- **35-01**: Delete custom model files and SAM2 pseudo-labeler (CLEAN-01 + CLEAN-02)
  ([`768516f`](https://github.com/tlancaster6/AquaPose/commit/768516f7f0eace404d3fc56635689f6af632dce9))

- **35-01**: Remove MOG2 detector and old training CLI commands (CLEAN-04 + CLEAN-05)
  ([`623b2c7`](https://github.com/tlancaster6/AquaPose/commit/623b2c70bccf057b4b61cc1c021a663cee814ebf))

- **35-02**: Correct planning documents to reflect CLEAN-03 semantics
  ([`bb265ca`](https://github.com/tlancaster6/AquaPose/commit/bb265ca55749b1ad06ec5b46136e55bcae5f0d37))

- **35-02**: Stub midline backends as no-ops and add config validation (CLEAN-03)
  ([`356022e`](https://github.com/tlancaster6/AquaPose/commit/356022e10be4386d898687e123d1b33e8c3bc7ca))

- **36-01**: Add --mode flag and generate_seg_dataset to build_yolo_training_data
  ([`ba6248a`](https://github.com/tlancaster6/AquaPose/commit/ba6248ab37cf485ab9ee1bf84e460901216f5f4f))

- **36-01**: Add TestFormatSegAnnotation and TestSegConverter test classes
  ([`a9defaf`](https://github.com/tlancaster6/AquaPose/commit/a9defaf97cf417ee3d0205a20da17f103e8e523e))

- **36-01**: Add yolo_seg/yolo_pose training wrappers and CLI commands
  ([`e7e41e2`](https://github.com/tlancaster6/AquaPose/commit/e7e41e2b729166d3c056cf0cde50b53011089412))

- **36-02**: Add seg/pose CLI help tests and update test_training_cli.py
  ([`7c2640e`](https://github.com/tlancaster6/AquaPose/commit/7c2640e8b8ead0207a1c1c078768a098ad6209a0))

- **37-02**: Implement PoseEstimationBackend with YOLO-pose inference and spline fitting
  ([`cc55961`](https://github.com/tlancaster6/AquaPose/commit/cc559610c4b93ec4ced136f1b8a0fbfd29d8ee27))

- **37-02**: Implement SegmentationBackend with YOLO-seg inference and skeletonization
  ([`f6b517c`](https://github.com/tlancaster6/AquaPose/commit/f6b517ca9705def48d76435b6d64d1fd54682ce3))

- **quick-12**: Implement YOLO training data builder core
  ([`a4d57ee`](https://github.com/tlancaster6/AquaPose/commit/a4d57ee6efa50aa65a15ac97a29401a78a081835))

- **quick-14**: Convert build script to emit Ultralytics-native NDJSON
  ([`f004338`](https://github.com/tlancaster6/AquaPose/commit/f0043388de44acab8157c0a8c3820a7e1502cdb5))

### Refactoring

- **33**: Remove torch parameter threading, simplify config extraction
  ([`d6e01cc`](https://github.com/tlancaster6/AquaPose/commit/d6e01cc9b119cd96fe363e7e0bcc54d55426b59f))

- **37-01**: Rename midline backends to segmentation/pose_estimation
  ([`6204312`](https://github.com/tlancaster6/AquaPose/commit/6204312c2591c86dc6e055a344832ab7fee96795))

- **quick-12**: Switch YOLO training data output from per-image txt to NDJSON
  ([`9c9769a`](https://github.com/tlancaster6/AquaPose/commit/9c9769af740380c28ec6c5b8034827a816fa3bd6))

- **quick-13**: Rewrite yolo_obb.py to use NDJSON pipeline via train_yolo_ndjson
  ([`cd3e90a`](https://github.com/tlancaster6/AquaPose/commit/cd3e90a913c4e901137b9b15a97aee6431f3f57c))

- **quick-14**: Simplify training wrappers to pass NDJSON directly to model.train()
  ([`6fddbf7`](https://github.com/tlancaster6/AquaPose/commit/6fddbf78715b621f2c72a41ea0b178ea115238e4))

- **training**: Extract shared COCO helpers and fix metric direction heuristic
  ([`b2ed161`](https://github.com/tlancaster6/AquaPose/commit/b2ed16144ac354f011b7a96a3d105d90e7d188a6))

- **training**: Extract shared NDJSON/YAML/training helpers into common.py
  ([`fba52c4`](https://github.com/tlancaster6/AquaPose/commit/fba52c44338f87953c01ad2357416bb63fd46ddf))

### Testing

- **30-01**: Add tests for strict reject, rename hints, and dataclass extensions
  ([`a2ed4e8`](https://github.com/tlancaster6/AquaPose/commit/a2ed4e8dca7dd352edc6911df3f63866e339b8f7))

- **30-02**: Add config promotion tests and CPU E2E test
  ([`11d0915`](https://github.com/tlancaster6/AquaPose/commit/11d091531c1f9382e348f98cc2a0dadb44dd2b1f))

- **33.1-01**: Add comprehensive tests for augmentation, visibility, and masked loss
  ([`b27b126`](https://github.com/tlancaster6/AquaPose/commit/b27b1264cd15b41ca76f5e43dd21e1de851252dd))

- **37-01**: Update tests to use new segmentation/pose_estimation backend names
  ([`f114194`](https://github.com/tlancaster6/AquaPose/commit/f114194e2655262a983f2fbb8748a83817cc85f2))

- **quick-12**: Add unit and integration tests for YOLO training data builder
  ([`847e6d7`](https://github.com/tlancaster6/AquaPose/commit/847e6d73b379d25496c1407c77ae3cf5888412f8))


## v1.1.0-dev.3 (2026-02-28)

### Bug Fixes

- Adaptive t_saturate for short runs and increase midline detection tolerance
  ([`2e825c8`](https://github.com/tlancaster6/AquaPose/commit/2e825c8875c128916cafef413321eabd0e7c1aca))

- Auto-generate LUTs on cache miss, fix Midline2D reconstruction
  ([`b413b51`](https://github.com/tlancaster6/AquaPose/commit/b413b5115d82ca391060f5c3f91d8c749a78582f))

- Make reconstruction and visualization honor configurable n_control_points
  ([`a9d61ca`](https://github.com/tlancaster6/AquaPose/commit/a9d61ca8d26b24eb11d4adaec7cd0f836b71ec9a))

- Pass stop_frame to TrackletTrailObserver via observer_factory
  ([`5923879`](https://github.com/tlancaster6/AquaPose/commit/59238795b777899c645ea231dad042c67076a368))

- Stabilize 3D animation axes and remove DiagnosticObserver from CLI modes
  ([`ab1cdd9`](https://github.com/tlancaster6/AquaPose/commit/ab1cdd9551818a9d07cf40676ecb9cf8b69de401))

- Suppress boxmot loguru spam in OcSortTracker._create_tracker
  ([`02f422a`](https://github.com/tlancaster6/AquaPose/commit/02f422a125d8fd1f226c8ef34a347111c875d23a))

- TrackletTrailObserver respects stop_frame parameter
  ([`272e221`](https://github.com/tlancaster6/AquaPose/commit/272e22161103c1618d52ab791d6e1c63eed02f77))

- Update stale docstring and requirements checkboxes for v2.1 completion
  ([`2de236b`](https://github.com/tlancaster6/AquaPose/commit/2de236b96a5bddd58122955d11f7a82a1a39635a))

- Update stale test mocks for spline knots/degree and warning message
  ([`4609295`](https://github.com/tlancaster6/AquaPose/commit/4609295ba424497abbe48ee2ca67a523c8cd4be6))

- Use white for midline head arrowhead to distinguish from body
  ([`7199dea`](https://github.com/tlancaster6/AquaPose/commit/7199deabe79c5ce377ad329911d20253e2c29a12))

- **26**: Revise plans based on checker feedback
  ([`12b7c47`](https://github.com/tlancaster6/AquaPose/commit/12b7c47aae68fe5ca4b4f79b31669a1919892bf6))

- **28**: Skip golden tests referencing deleted v1.0 modules
  ([`4396f71`](https://github.com/tlancaster6/AquaPose/commit/4396f71ce86c7697fede1a51ed68b8d278b6d6a0))

- **e2e**: Remove invalid device kwarg from MidlineConfig, skip spline test without LUTs
  ([`0b6a5fa`](https://github.com/tlancaster6/AquaPose/commit/0b6a5faa33066f663f1fe130596bad2f761f2032))

### Chores

- Complete v2.1 Identity milestone
  ([`a38512d`](https://github.com/tlancaster6/AquaPose/commit/a38512db354d516d8a93e3b7dd1bbd621ed5d2ad))

- Remove 10 stale inbox documents superseded by GUIDEBOOK and shipped code
  ([`ce84dc8`](https://github.com/tlancaster6/AquaPose/commit/ce84dc862bf28ac6af5b0d8b57c503c53ddb9a52))

### Documentation

- Capture todo - Add per-stage diagnostic visualizations
  ([`72af988`](https://github.com/tlancaster6/AquaPose/commit/72af988966493f789c8c0bdbd70614ed905b637c))

- Capture todo - Add project_dir base path to pipeline config
  ([`95fa900`](https://github.com/tlancaster6/AquaPose/commit/95fa9003a99d844fba9292124daa272ac80965a3))

- Capture todo - Audit and update GUIDEBOOK.md
  ([`52a2044`](https://github.com/tlancaster6/AquaPose/commit/52a2044a9cc83778492ad4ed4e634f6ad24af941))

- Capture todo - Clean up and reorganize pipeline config schema
  ([`f3e3480`](https://github.com/tlancaster6/AquaPose/commit/f3e3480b7495587afb4f67587afb303cc3ba6092))

- Capture todo - Improve midline orientation logic for segment-then-extract backend
  ([`4944b10`](https://github.com/tlancaster6/AquaPose/commit/4944b107ae7bb3173aaae6ec97985aa2710ee1e7))

- Capture todo - Regenerate golden regression test data for v2.1
  ([`1aec990`](https://github.com/tlancaster6/AquaPose/commit/1aec99072094560a79ca547d864b16c69aa85e96))

- Close todo - Windowed velocity smoothing (superseded by OC-SORT)
  ([`064f6b5`](https://github.com/tlancaster6/AquaPose/commit/064f6b54e5bdebb1197ee292e39c0e7200b92380))

- Create milestone v2.1 roadmap (6 phases)
  ([`39d4949`](https://github.com/tlancaster6/AquaPose/commit/39d4949baa4c6b92bca6dad04702c5adb75a28b9))

- Define milestone v2.1 requirements
  ([`c30150b`](https://github.com/tlancaster6/AquaPose/commit/c30150b4cd955ddc3c3cb434a98d95b49af4698c))

- Refine v2.1 prospective, add association design spec, update guidebook
  ([`1812095`](https://github.com/tlancaster6/AquaPose/commit/18120954bef7ee6a03a6e1d613295eedd7c8b104))

- Start milestone v2.1 Identity
  ([`b4c488e`](https://github.com/tlancaster6/AquaPose/commit/b4c488edd6861590d88678ab7999366ca9239a5b))

- Update Phase 22 plans with CarryForward and test_writer.py additions
  ([`3c26ff1`](https://github.com/tlancaster6/AquaPose/commit/3c26ff145ef50ded24e044675580cc043c36f2a5))

- V2.1 milestone audit — all requirements satisfied, tech debt only
  ([`00675df`](https://github.com/tlancaster6/AquaPose/commit/00675dfe44c4a54090a47e474885de62b20c2ef4))

- **22**: Capture phase context
  ([`ec247c2`](https://github.com/tlancaster6/AquaPose/commit/ec247c230c642af20f73ddc3c4b71a74498af1e2))

- **22**: Create phase plan for pipeline scaffolding
  ([`31e8afb`](https://github.com/tlancaster6/AquaPose/commit/31e8afbedd12cdb7cb0cdbc40666247896ff81b4))

- **22-01**: Complete domain types and legacy deletion plan
  ([`25c6d7b`](https://github.com/tlancaster6/AquaPose/commit/25c6d7bea4b885cf51831504f03d26a1adc0d55a))

- **22-02**: Complete stub stages and pipeline rewire plan
  ([`057f042`](https://github.com/tlancaster6/AquaPose/commit/057f0427d566f20f4d7e00fa40fbd9256d10d8de))

- **23**: Capture phase context
  ([`1d706db`](https://github.com/tlancaster6/AquaPose/commit/1d706dbac4770daeeb37448c007142104079cb1e))

- **23-01**: Complete forward LUT plan - SUMMARY.md, STATE.md, ROADMAP.md
  ([`18d674d`](https://github.com/tlancaster6/AquaPose/commit/18d674deb1d63d4fccff8d76b4459a0570c8b119))

- **23-02**: Complete InverseLUT plan - SUMMARY.md, STATE.md, ROADMAP.md
  ([`1b286ff`](https://github.com/tlancaster6/AquaPose/commit/1b286ff00ea3fb67abc6bfedbf9a9b2e4475c14b))

- **24**: Capture phase context
  ([`f057d22`](https://github.com/tlancaster6/AquaPose/commit/f057d22d0d7e2ef7753febb7dc26eec17fc764e3))

- **24**: Create phase plan for per-camera 2D tracking
  ([`bc4b724`](https://github.com/tlancaster6/AquaPose/commit/bc4b7244b0b4792e6e5d86a83de3761c15df5a0b))

- **24-01**: Complete per-camera 2D tracking plan — SUMMARY.md, STATE.md, ROADMAP.md
  ([`58391a1`](https://github.com/tlancaster6/AquaPose/commit/58391a16349b491b0fa8a7388b44564b903ad359))

- **25**: Add phase verification
  ([`11b24cd`](https://github.com/tlancaster6/AquaPose/commit/11b24cd40c690366ddbda07be78249642c12b116))

- **25**: Capture phase context
  ([`2ff2fc3`](https://github.com/tlancaster6/AquaPose/commit/2ff2fc32a3d10189684e2cd953270e64375c07d0))

- **25**: Capture phase context
  ([`99d8057`](https://github.com/tlancaster6/AquaPose/commit/99d80572d3bef6e79324d62a40f9dd41221842fb))

- **25**: Create phase plans for association scoring and clustering
  ([`495ce1b`](https://github.com/tlancaster6/AquaPose/commit/495ce1b79470e4ea71fb1731621c6281b499ab68))

- **25-01**: Complete pairwise scoring plan — SUMMARY.md, ROADMAP.md
  ([`6c500c7`](https://github.com/tlancaster6/AquaPose/commit/6c500c7eb228e56d8bce52fc2f7f97a95e14961c))

- **26**: Add phase verification
  ([`2d27723`](https://github.com/tlancaster6/AquaPose/commit/2d2772374661691aef0337388146f4bc1849c84d))

- **26**: Capture phase context
  ([`fbe4587`](https://github.com/tlancaster6/AquaPose/commit/fbe45873a26064077d3d708a3b4dbb2b10d96d4c))

- **26-01**: Complete cluster refinement plan -- SUMMARY.md, ROADMAP.md
  ([`7f659fe`](https://github.com/tlancaster6/AquaPose/commit/7f659fe117f66b722d816c594a3cdf60d3a3fa4d))

- **26-02**: Complete midline orientation plan -- SUMMARY.md, ROADMAP.md
  ([`8eb04fe`](https://github.com/tlancaster6/AquaPose/commit/8eb04fefbc5d5047d3f60950c69a1f7d1c833bb4))

- **26-03**: Add summary, mark phase 26 complete in ROADMAP
  ([`64bcd74`](https://github.com/tlancaster6/AquaPose/commit/64bcd74ebe1ced61d38a81828a44b96e061e64d5))

- **27**: Add phase verification, mark phase 27 complete
  ([`734cb6e`](https://github.com/tlancaster6/AquaPose/commit/734cb6ed3b97d0cb51ec706ee2becd0f7e60598d))

- **27**: Capture phase context
  ([`74f3c27`](https://github.com/tlancaster6/AquaPose/commit/74f3c27d809b470f27405f65f3f8ddc368b777ba))

- **27**: Create phase plan
  ([`ec3cd4e`](https://github.com/tlancaster6/AquaPose/commit/ec3cd4e0f72024449a8b10f2d4be7cdc6696142b))

- **27-01**: Complete TrackletTrailObserver plan -- SUMMARY.md, STATE.md, ROADMAP.md
  ([`fac098d`](https://github.com/tlancaster6/AquaPose/commit/fac098d6450d838c3f61191a4ccce253d396c0e0))

- **28**: Add synthetic data approach to context
  ([`503352c`](https://github.com/tlancaster6/AquaPose/commit/503352c74c55bcb914467fd9861ffa15c398a4ef))

- **28**: Capture phase context
  ([`4a9eb4f`](https://github.com/tlancaster6/AquaPose/commit/4a9eb4fc03cf96223a7838cdb24ab44842438c30))

- **28**: Create phase plan for e2e testing
  ([`1354cb5`](https://github.com/tlancaster6/AquaPose/commit/1354cb514b9d3f1c57a1611e52a4b4fff2b24a91))

- **28-01**: Complete e2e test plan -- checkpoint at human-verify task
  ([`0e7c632`](https://github.com/tlancaster6/AquaPose/commit/0e7c632864379bf5ac174f856b6ed07fa4f72758))

- **phase-22**: Complete phase execution and verification
  ([`0d151f7`](https://github.com/tlancaster6/AquaPose/commit/0d151f7af58b353f1698f8940dfb22e5b10ffade))

- **phase-23**: Complete phase execution and verification
  ([`8a41673`](https://github.com/tlancaster6/AquaPose/commit/8a4167364c120deb191dc7a0dd1d8ebebdce3b31))

- **phase-24**: Complete phase execution
  ([`9016bbe`](https://github.com/tlancaster6/AquaPose/commit/9016bbebc4fa96bcfdf8abd00710c63320644c94))

- **state**: Record phase 28 context session
  ([`1dcc50e`](https://github.com/tlancaster6/AquaPose/commit/1dcc50ea745746d61fba794b386c458bad497821))

### Features

- Add --stop-after CLI option for partial pipeline runs
  ([`820fd6f`](https://github.com/tlancaster6/AquaPose/commit/820fd6ff15267e4fd780e56d30d83d257e9b704d))

- Add top-level n_animals config field with propagation to sub-configs
  ([`40c62df`](https://github.com/tlancaster6/AquaPose/commit/40c62df198992f33832dd8a8c1a887ff91ad2b76))

- Draw head-direction arrowhead on midline overlays
  ([`7481f0c`](https://github.com/tlancaster6/AquaPose/commit/7481f0c1450288b01794b396143112c4c6571cb0))

- **22-01**: Define Tracklet2D, TrackletGroup, CarryForward domain types
  ([`3a014fa`](https://github.com/tlancaster6/AquaPose/commit/3a014fa19b15f61fc8dc32db9be39fac6872edcf))

- **22-01**: Delete legacy tracking/association code and update consumers
  ([`634efc7`](https://github.com/tlancaster6/AquaPose/commit/634efc706c98006be5bdf9172f2d9db77b3e2ca5))

- **22-02**: Adapt all affected tests to new 5-stage pipeline structure
  ([`5475a0a`](https://github.com/tlancaster6/AquaPose/commit/5475a0a53c1e32ae6a4c4c713a185d200473c461))

- **22-02**: Add TrackingStubStage/AssociationStubStage, rewire build_stages() to 5-stage order
  ([`6b7a2c2`](https://github.com/tlancaster6/AquaPose/commit/6b7a2c2426a1819358d3472e60932b26afd6ef66))

- **23-01**: Add ForwardLUT class, generate_forward_luts(), LutConfig
  ([`afde176`](https://github.com/tlancaster6/AquaPose/commit/afde1762b7b7a9fb0bff4b0cb4c11c2fb1f0a88b))

- **23-02**: Implement InverseLUT with voxel grid, overlap graph, ghost-point lookup
  ([`1079dca`](https://github.com/tlancaster6/AquaPose/commit/1079dcacfd524639c241c1c14f180e23c2ef0ac1))

- **24-01**: Add boxmot dependency and OcSortTracker wrapper
  ([`04b36cf`](https://github.com/tlancaster6/AquaPose/commit/04b36cf16aaedbcee92c0ac0c07de7bdf16eacc0))

- **24-01**: Implement TrackingStage, expand TrackingConfig, rewire pipeline
  ([`e303255`](https://github.com/tlancaster6/AquaPose/commit/e303255cd774dbe11a7384d91f38e793a89cc1a3))

- **25-01**: Implement pairwise cross-camera tracklet affinity scoring
  ([`670ab3b`](https://github.com/tlancaster6/AquaPose/commit/670ab3b8a61584ca6ff921d30b3788b151eb2568))

- **25-02**: Implement Leiden clustering, fragment merging, and AssociationStage
  ([`d9cbab7`](https://github.com/tlancaster6/AquaPose/commit/d9cbab707dabf37da9084374b16685784c82b57a))

- **26-01**: Add refinement config fields and per_frame_confidence to TrackletGroup
  ([`b3285da`](https://github.com/tlancaster6/AquaPose/commit/b3285da226a6fc1803ed583b7b32bde642e28075))

- **26-01**: Implement refine_clusters() with eviction and wire into AssociationStage
  ([`cd783de`](https://github.com/tlancaster6/AquaPose/commit/cd783deb2bdd8580bae6377662a2b3bad24df3eb))

- **26-02**: Implement head-tail orientation resolver with 3-signal combination
  ([`dd4f553`](https://github.com/tlancaster6/AquaPose/commit/dd4f553b45126269204cf7cbdebbbbee391a061e))

- **26-02**: Update MidlineStage for tracklet-group filtering and orientation resolution
  ([`8546017`](https://github.com/tlancaster6/AquaPose/commit/85460173744f04a8bdee2619fdf10ff00bc7de34))

- **26-03**: Rewrite ReconstructionStage for tracklet-group camera membership and fish-first HDF5
  ([`fda7fae`](https://github.com/tlancaster6/AquaPose/commit/fda7fae15d06eab07270c90e9163357c0df8311d))

- **27-01**: Implement TrackletTrailObserver with per-camera trails and association mosaic
  ([`8354744`](https://github.com/tlancaster6/AquaPose/commit/83547448dd736653602e4b812c947dff670b2d70))

- **27-01**: Wire TrackletTrailObserver into factory, exports, and unit tests
  ([`b00d7ac`](https://github.com/tlancaster6/AquaPose/commit/b00d7ac77abd5a4686c2cda1c343253e127a0cd3))

- **28-01**: Rewrite e2e tests and fix synthetic fish placement
  ([`b7adf90`](https://github.com/tlancaster6/AquaPose/commit/b7adf90d69b3f2652fceded5e752f31983f9517f))

### Refactoring

- **e2e**: Remove persistent output, use tmp_path for all artifacts
  ([`64a7f78`](https://github.com/tlancaster6/AquaPose/commit/64a7f782b805d96fa3c54eea6a3589a96404787c))

- **e2e**: Resolve test data from ~/aquapose/testing/
  ([`182b0ae`](https://github.com/tlancaster6/AquaPose/commit/182b0aeb342dc9939fb8b81c44597972ca307f86))

### Testing

- **23-01**: Add unit tests for ForwardLUT generation, interpolation, serialization
  ([`ffa9b95`](https://github.com/tlancaster6/AquaPose/commit/ffa9b958b85e6f9346dc624db4d302d59d78cda4))

- **23-02**: Add 8 InverseLUT unit tests covering voxel grid, visibility, overlap graph,
  serialization
  ([`d37bf80`](https://github.com/tlancaster6/AquaPose/commit/d37bf808687849c5fd559832e6c57ee00ab105bc))


## v1.1.0-dev.2 (2026-02-27)

### Bug Fixes

- **16**: Accept device kwarg in YOLOBackend to match build_stages() contract
  ([`e0964d8`](https://github.com/tlancaster6/AquaPose/commit/e0964d8ebc6333b06ec15ab2f3c168e32e72e59b))

- **16**: Adjust regression test tolerances for CUDA non-determinism
  ([`214a3c6`](https://github.com/tlancaster6/AquaPose/commit/214a3c6d377c8205891ec0ebd4da2f000fa735c6))

- **16**: Regenerate golden data with v2.0 PosePipeline and relax determinism tolerance
  ([`1b4587a`](https://github.com/tlancaster6/AquaPose/commit/1b4587abaaf528573a2a82021b1c2df2ed4a8773))

- **16**: Use correct RefractiveProjectionModel constructor in 3 backends
  ([`2f67c93`](https://github.com/tlancaster6/AquaPose/commit/2f67c931b5c88c22177e9d120e23651231fa572a))

- **16**: Widen regression tolerances for observed CUDA jitter
  ([`9f663c1`](https://github.com/tlancaster6/AquaPose/commit/9f663c16df1a2cb972497770aae61fd2c5e714d9))

- **19**: Human verification fixes — CLI traceback, type coercion, smoke test logging, dead code
  audit
  ([`c5e230d`](https://github.com/tlancaster6/AquaPose/commit/c5e230db04ec52069175046f8d54356a02853173))

- **20-05**: Replace hardcoded paths in regression conftest with env vars
  ([`3fa5c2f`](https://github.com/tlancaster6/AquaPose/commit/3fa5c2f05f5fefb504ea5e1e2425e28485cdce45))

- **cli**: Respect mode from config YAML when --mode not passed on CLI
  ([`8c47fe9`](https://github.com/tlancaster6/AquaPose/commit/8c47fe9bc655f5f7cb1dcfffd5f66110fe450b91))

- **overlay**: Build per-camera RefractiveProjectionModel instead of passing CalibrationData
  ([`0882b96`](https://github.com/tlancaster6/AquaPose/commit/0882b96a7c50dc5805f09508cad480a295ce0324))

- **overlay**: Support timestamped video filenames via prefix glob fallback
  ([`988c3fb`](https://github.com/tlancaster6/AquaPose/commit/988c3fb3617084a2d26e7beb1a57a49be7095af4))

- **overlay**: Use VideoSet for undistorted frame reading
  ([`06df1ae`](https://github.com/tlancaster6/AquaPose/commit/06df1aeaa2da82c278fe8bf0569e67e467a3cff9))

- **tracking**: Remove experimental near-claim penalty ghost suppression
  ([`2187512`](https://github.com/tlancaster6/AquaPose/commit/2187512a9a4d22b9c14af2a21b070748e3535cf9))

### Chores

- Archive v2.0 phase directories to milestones/v2.0-phases/
  ([`82d8d9b`](https://github.com/tlancaster6/AquaPose/commit/82d8d9b7485b3418e388cd4cd815e6527c991583))

- Complete v2.0 Alpha milestone
  ([`e4d6f12`](https://github.com/tlancaster6/AquaPose/commit/e4d6f124dfb36f88140e4f5965834a5827640201))

- Deleting legacy scripts
  ([`ddc2bd0`](https://github.com/tlancaster6/AquaPose/commit/ddc2bd02fc2a94b2a0a20fbccac98c6681383a13))

- **14.1-01**: Align ROADMAP and REQUIREMENTS to 5-stage pipeline model
  ([`428889b`](https://github.com/tlancaster6/AquaPose/commit/428889b00a8ac1d5889f51e7f0f14515d17385c3))

- **14.1-01**: Delete redundant inbox documents superseded by guidebook
  ([`ac41419`](https://github.com/tlancaster6/AquaPose/commit/ac41419261475d2cf7ce374436a554aaa44236e1))

- **16-02**: Archive legacy diagnostic scripts to scripts/legacy/
  ([`2569ce8`](https://github.com/tlancaster6/AquaPose/commit/2569ce87640313bbebd81de5e226fa7042ce66bd))

- **20-02**: Delete dead modules, orphaned tests, and legacy scripts
  ([`fa21b8b`](https://github.com/tlancaster6/AquaPose/commit/fa21b8b3ec390d8b80030c030ccf5591d5bb824a))

- **21-01**: Mark CLI-01 through CLI-05 complete in REQUIREMENTS.md
  ([`d994268`](https://github.com/tlancaster6/AquaPose/commit/d994268065d8c0ba89ae74216657f389df1864af))

### Code Style

- **20-01**: Apply ruff formatting to post-import-boundary-fix files
  ([`e29b9cc`](https://github.com/tlancaster6/AquaPose/commit/e29b9cc6c55d0dff30a862d9bd5653078be89240))

### Documentation

- Add phase 16 pickup notes for next session
  ([`df27799`](https://github.com/tlancaster6/AquaPose/commit/df2779994590e6d3593dab4a846cea786756ea2b))

- Commit accumulated planning updates from prior sessions
  ([`04c070e`](https://github.com/tlancaster6/AquaPose/commit/04c070e1a11f6a55d87e531819028b495ca7b412))

- Commit debug session notes from prior sessions
  ([`f5deb61`](https://github.com/tlancaster6/AquaPose/commit/f5deb6151a182eed3d98f3b3c4610fdfa593acf7))

- Commit missed planning files for phases 17, 19, 20, 21
  ([`37d1eb3`](https://github.com/tlancaster6/AquaPose/commit/37d1eb31a00aff4b7cfb9ab65fb36ed6d1945db9))

- Create milestone v2.0 roadmap (6 phases)
  ([`ccc0707`](https://github.com/tlancaster6/AquaPose/commit/ccc0707fed94bdcafff86279711213f2d2d2ad4f))

- Create v2.0 milestone audit report
  ([`bd327bd`](https://github.com/tlancaster6/AquaPose/commit/bd327bde3d01b8fc4e60b54428eb49df5eb4337c))

- Define milestone v2.0 requirements
  ([`8391ec6`](https://github.com/tlancaster6/AquaPose/commit/8391ec6b69c88d84883b4fc71622e9e34407462c))

- Start milestone v2.0 Alpha
  ([`b8bc331`](https://github.com/tlancaster6/AquaPose/commit/b8bc3315bea7530c06fe83132c94be6dde622fc2))

- **13**: Capture phase context
  ([`e5d16d5`](https://github.com/tlancaster6/AquaPose/commit/e5d16d56d94dc0a3d02f3db861b3e379b8392ddc))

- **13-01**: Complete Stage Protocol and PipelineContext plan
  ([`2d1d5f6`](https://github.com/tlancaster6/AquaPose/commit/2d1d5f6202c1a2d387ef0c9b7d29a2cc223e3b0f))

- **13-02**: Complete config hierarchy plan
  ([`4f5cd34`](https://github.com/tlancaster6/AquaPose/commit/4f5cd341a3b8e007969a505b90952e08857d4799))

- **13-03**: Complete event system and observer protocol plan
  ([`6278165`](https://github.com/tlancaster6/AquaPose/commit/6278165fa11f23d97d2ba6375796b5676f37b118))

- **13-engine-core**: Complete plan 04 - PosePipeline orchestrator
  ([`52a5728`](https://github.com/tlancaster6/AquaPose/commit/52a57284a445cb47961ac128e507738dfb846608))

- **13-engine-core**: Create phase plan
  ([`2b38e04`](https://github.com/tlancaster6/AquaPose/commit/2b38e04e4797e679f519b4839ae8166062dd6bee))

- **14**: Capture phase context
  ([`e57f00b`](https://github.com/tlancaster6/AquaPose/commit/e57f00b7dd04e51faba448398f689ef5792ee9db))

- **14**: Create phase plan for golden data and verification framework
  ([`4501682`](https://github.com/tlancaster6/AquaPose/commit/45016820d4ea7a696fe48ef94effec8eaee0d8e6))

- **14-01**: Complete golden data generation script plan
  ([`fb2c96b`](https://github.com/tlancaster6/AquaPose/commit/fb2c96b0e1e287a46117ad53195aa7fb13d9e59f))

- **14-02**: Complete golden regression harness plan — Phase 14 done
  ([`346002b`](https://github.com/tlancaster6/AquaPose/commit/346002b63acae236657c119896fc181cdd5b7eec))

- **14.1**: Capture phase context
  ([`2146799`](https://github.com/tlancaster6/AquaPose/commit/21467998d4a8511f140bab2ae13191aa7356610e))

- **14.1-01**: Complete planning doc alignment plan — 5-stage model established
  ([`287c9c1`](https://github.com/tlancaster6/AquaPose/commit/287c9c1f5c323751ed3250e8648a58d6d944e33f))

- **14.1-02**: Complete engine 5-stage alignment plan — Phase 14.1 done
  ([`00287db`](https://github.com/tlancaster6/AquaPose/commit/00287db567e76953d6c34924cc8ccc7a55335c3c))

- **14.1-02**: Update golden test harness to clarify v1.0-to-5-stage mapping
  ([`5399cfd`](https://github.com/tlancaster6/AquaPose/commit/5399cfde5acddaec9a98218dff1ccbb12f17b47a))

- **15**: Capture phase context
  ([`8e44efb`](https://github.com/tlancaster6/AquaPose/commit/8e44efb7a8874f793bbd2eb0d7ae61e2c19321d4))

- **15**: Create 5 stage migration plans
  ([`a4f6eb1`](https://github.com/tlancaster6/AquaPose/commit/a4f6eb1643cdf94a83ae7bc1ecf0c3a5f939c10f))

- **15-01**: Complete DetectionStage migration plan — Phase 15 Plan 01 done
  ([`b073b5a`](https://github.com/tlancaster6/AquaPose/commit/b073b5a5884a3edd410d4e343c964654d16a4430))

- **15-02**: Complete MidlineStage migration plan — Phase 15 Plan 02 done
  ([`b022e18`](https://github.com/tlancaster6/AquaPose/commit/b022e18b8ba3755db97ff2d07c418de0651db806))

- **15-03**: Complete AssociationStage migration plan — Phase 15 Plan 03 done
  ([`1ebacb6`](https://github.com/tlancaster6/AquaPose/commit/1ebacb644988ee2a8a77938006ab88e59f119dc6))

- **15-03**: Mark STG-03 requirement complete in REQUIREMENTS.md
  ([`c598f09`](https://github.com/tlancaster6/AquaPose/commit/c598f09af288d227d9fdde5baeabbd6ae84231b1))

- **15-04**: Complete TrackingStage migration plan — Phase 15 Plan 04 done
  ([`e6d714c`](https://github.com/tlancaster6/AquaPose/commit/e6d714c4fa512eb992bd54c78267125b42f418cf))

- **15-04**: Mark STG-04 requirement complete in REQUIREMENTS.md
  ([`c51a702`](https://github.com/tlancaster6/AquaPose/commit/c51a7027a4621879b84bc6c8913b6b6be01203b3))

- **15-05**: Add self-check results to SUMMARY.md
  ([`6bdea22`](https://github.com/tlancaster6/AquaPose/commit/6bdea225c24482cf56fe446ac554dfb3da9c724c))

- **15-05**: Complete ReconstructionStage migration plan — Phase 15 done
  ([`57e6c12`](https://github.com/tlancaster6/AquaPose/commit/57e6c12cf4c8e2b454f84129fe034b673d52b49d))

- **16**: Add phase verification report
  ([`d672bf1`](https://github.com/tlancaster6/AquaPose/commit/d672bf12a42224d51fd377a69486b42387cd78e7))

- **16**: Capture phase context
  ([`3966c7e`](https://github.com/tlancaster6/AquaPose/commit/3966c7ed2f0b6e7207aa24201e2c4b818ddc5071))

- **16**: Plan phase — 2 plans in 2 waves
  ([`dad385c`](https://github.com/tlancaster6/AquaPose/commit/dad385c6bd9404d9e4510b217c1cb98f9048715c))

- **16-01**: Add self-check results to SUMMARY.md
  ([`98050e7`](https://github.com/tlancaster6/AquaPose/commit/98050e7384f84626783dc377091067778f640b77))

- **16-01**: Complete regression test suite plan — Phase 16 Plan 1
  ([`823c006`](https://github.com/tlancaster6/AquaPose/commit/823c006b2bc4f95d7aa4f05d5032878dd9fe319c))

- **16-02**: Complete legacy script archive plan — Phase 16 done
  ([`0d95582`](https://github.com/tlancaster6/AquaPose/commit/0d9558260579c9b1e84903c1341e3480d0d48688))

- **17**: Add plan summaries for all 5 observer plans
  ([`7d2931b`](https://github.com/tlancaster6/AquaPose/commit/7d2931be50370d3966deaaa2f2a84f60f06ff8db))

- **17**: Capture phase context
  ([`6a3c573`](https://github.com/tlancaster6/AquaPose/commit/6a3c573d22f57ae712c0ca84463d713561ef52b8))

- **18**: Add plan summaries for all 3 CLI and execution mode plans
  ([`14c15ff`](https://github.com/tlancaster6/AquaPose/commit/14c15ffe1862d80394d2474c9491bdfa3fcf0e8e))

- **18**: Capture phase context
  ([`1a62aa3`](https://github.com/tlancaster6/AquaPose/commit/1a62aa3dc1bcffaffa26b7750671d78a160ca4b1))

- **18**: Create phase plans for CLI and Execution Modes
  ([`bf3c3bb`](https://github.com/tlancaster6/AquaPose/commit/bf3c3bb443241f93b379831222ba8db3a487bf9c))

- **19**: Capture phase context
  ([`a48b891`](https://github.com/tlancaster6/AquaPose/commit/a48b891c67135519fa8921bf8e7e6ca916013c0b))

- **19**: Create phase plan for alpha refactor audit
  ([`c7d9c15`](https://github.com/tlancaster6/AquaPose/commit/c7d9c15f622cae401de0e061c3bfbd966379a55b))

- **19-01**: Complete import boundary checker plan
  ([`87cf8ff`](https://github.com/tlancaster6/AquaPose/commit/87cf8ff0b84bb986841b2cee2b07f0232c942536))

- **19-02**: Complete smoke test plan summary and state update
  ([`ee65cb0`](https://github.com/tlancaster6/AquaPose/commit/ee65cb0a1eda2932cc2276265323bc14ef8f2c3f))

- **19-03**: Complete alpha refactor audit plan
  ([`72bbd0a`](https://github.com/tlancaster6/AquaPose/commit/72bbd0a6e06bb352fecc0dddfe38a402c32d6fb6))

- **19-03**: Produce comprehensive alpha refactor audit report
  ([`e1bcb22`](https://github.com/tlancaster6/AquaPose/commit/e1bcb22929a2a840e5df9a66e71d947eb1a6b034))

- **19-04**: Triage all 7 Phase 15 bug ledger items
  ([`c6caae9`](https://github.com/tlancaster6/AquaPose/commit/c6caae9ad75da81cdfe35785fffb53327a743d55))

- **20**: Capture phase context
  ([`1ce4a5d`](https://github.com/tlancaster6/AquaPose/commit/1ce4a5db1a1c05a2577905cb474d9c8c14b3670d))

- **20**: Create phase plan - 5 plans in 2 waves
  ([`f908bec`](https://github.com/tlancaster6/AquaPose/commit/f908becd633fb6702860877fbe302b3a972ada62))

- **20-01**: Complete plan - move PipelineContext/Stage to core/context
  ([`855d4a2`](https://github.com/tlancaster6/AquaPose/commit/855d4a209eaaf03fc7ff87953b0addd54ee22c81))

- **20-02**: Complete dead module deletion plan summary and state update
  ([`a500a20`](https://github.com/tlancaster6/AquaPose/commit/a500a208dc64bed0ee2963afb63bd2ca599f96ee))

- **20-03**: Complete plan - skip_camera removal verified + build_observers extracted to engine
  ([`ee198ae`](https://github.com/tlancaster6/AquaPose/commit/ee198aef33d4e24d4f539ba12000e94be23c9e38))

- **20-04**: Complete Stage 3/4 coupling fix plan — TrackingStage consumes bundles, 514 tests pass
  ([`c6f033a`](https://github.com/tlancaster6/AquaPose/commit/c6f033a0a43e5a1a99b37407d875223275e9d2b0))

- **20-05**: Complete remaining audit remediations plan — camera discovery, diagnostics split,
  regression env vars
  ([`5a80c44`](https://github.com/tlancaster6/AquaPose/commit/5a80c44427c0d17ee1d4b67e497c432393fa3203))

- **21**: Capture phase context
  ([`bbfa74a`](https://github.com/tlancaster6/AquaPose/commit/bbfa74acae7070b0f1068269d99459263179d21b))

- **21-01**: Complete retrospective plan — v2.0 Alpha narrative + requirements fix
  ([`7862283`](https://github.com/tlancaster6/AquaPose/commit/786228340715b0f2f58a67c3b1364999fc787f30))

- **21-01**: Write v2.0 Alpha retrospective document
  ([`5b10ab4`](https://github.com/tlancaster6/AquaPose/commit/5b10ab43429e2da2bb92a5c51bc1ee4eecf1bcde))

- **21-02**: Complete prospective plan — v2.1 requirements seed ready for /gsd:new-milestone
  ([`a8f14c2`](https://github.com/tlancaster6/AquaPose/commit/a8f14c23b89fb1d8f8a40cd83a0c8eab81c24885))

- **21-02**: Write v2.1 prospective document seeding next milestone requirements
  ([`5fa8e1e`](https://github.com/tlancaster6/AquaPose/commit/5fa8e1ed247d05e7e4b193d7d384dcec5e450b57))

- **phase-13**: Complete phase execution and verification
  ([`14ec545`](https://github.com/tlancaster6/AquaPose/commit/14ec545da52eb0b3f855c573f785efde480b3adc))

- **phase-14**: Complete phase execution and verification
  ([`5cfaead`](https://github.com/tlancaster6/AquaPose/commit/5cfaeadfa44205d093faa048b440a279daf2aab5))

- **phase-14.1**: Complete phase execution and verification
  ([`c4dd836`](https://github.com/tlancaster6/AquaPose/commit/c4dd836fd4aea40275abcb70e64c13a3db889c03))

- **phase-15**: Complete phase execution and verification
  ([`9cb5440`](https://github.com/tlancaster6/AquaPose/commit/9cb544099f26f132679c792fb569c6836a2be39b))

- **phase-16**: Mark phase complete in roadmap and state
  ([`e5bba86`](https://github.com/tlancaster6/AquaPose/commit/e5bba86d68ce6c1ce4ff001e843e2bd07295bd47))

- **phase-17**: Complete phase execution and verification
  ([`1b04327`](https://github.com/tlancaster6/AquaPose/commit/1b04327dbe9d04ec6329c740bfeba4347a02263c))

- **phase-18**: Complete phase execution and verification
  ([`bd95881`](https://github.com/tlancaster6/AquaPose/commit/bd958815a6871a7181203055aa5e768161d319e1))

- **phase-19**: Complete phase execution and verification
  ([`f2dc4c7`](https://github.com/tlancaster6/AquaPose/commit/f2dc4c7dc88efc229c552fa8c51657e6881e7d11))

- **phase-20**: Complete phase execution and verification
  ([`f14ca59`](https://github.com/tlancaster6/AquaPose/commit/f14ca59b1459577337abb9389727b343615ad4f8))

- **phase-21**: Complete phase execution and verification
  ([`4efb18e`](https://github.com/tlancaster6/AquaPose/commit/4efb18e33aaf5959c7b956f9f98121ff64873357))

- **quick-9**: Add init-config CLI command
  ([`d861067`](https://github.com/tlancaster6/AquaPose/commit/d86106720de2e575c0cc72ea87c0330eab95da1f))

- **quick-9**: Complete init-config CLI subcommand plan
  ([`54941d7`](https://github.com/tlancaster6/AquaPose/commit/54941d74f9f217b0439ca49fdb380803a8deba0f))

- **state**: Record phase 14.1 context session
  ([`d22b42d`](https://github.com/tlancaster6/AquaPose/commit/d22b42d9294a486f133b4f56c956a2df13e3cf8a))

- **state**: Record phase 19 context session
  ([`ae17c06`](https://github.com/tlancaster6/AquaPose/commit/ae17c0665e0b50b91caee526335ca5510bd9481c))

### Features

- **13-01**: Create engine package with Stage Protocol and PipelineContext
  ([`5047a49`](https://github.com/tlancaster6/AquaPose/commit/5047a49c1955f8a932eba8c2345b20721fb471f5))

- **13-02**: Implement frozen config dataclass hierarchy with YAML and CLI overrides
  ([`14245bf`](https://github.com/tlancaster6/AquaPose/commit/14245bfa22db4da417e7365f2e260326134da6c0))

- **13-02**: Write unit tests for config defaults, overrides, freeze, and serialization
  ([`4ff6c06`](https://github.com/tlancaster6/AquaPose/commit/4ff6c06e3ebdc3c95258a5611650f9205e8cf5eb))

- **13-engine-core**: Implement PosePipeline orchestrator skeleton
  ([`9ffe5fe`](https://github.com/tlancaster6/AquaPose/commit/9ffe5fec06b506bf10018f52b6eb4cfe8a0da18f))

- **14-01**: Add golden data generation script and tests/golden/ directory
  ([`ed79f27`](https://github.com/tlancaster6/AquaPose/commit/ed79f27019a3d0aa88d2871aa8362e923cde0216))

- **14-02**: Add golden test package init and session-scoped fixtures
  ([`f9aadbc`](https://github.com/tlancaster6/AquaPose/commit/f9aadbc4c47d3796f1af5fb127775662f6c73aa9))

- **14-02**: Add stage interface test harness with 9 golden regression tests
  ([`4085eb7`](https://github.com/tlancaster6/AquaPose/commit/4085eb7ed5116d606380c97c6bd2034f63036ae3))

- **14.1-02**: Update engine to 5-stage model — rename configs, update PipelineContext
  ([`ced4695`](https://github.com/tlancaster6/AquaPose/commit/ced4695bcf2e015d197aad508a8896c28775755f))

- **15-01**: Create core/detection/ module with DetectionStage and YOLO backend
  ([`65135fc`](https://github.com/tlancaster6/AquaPose/commit/65135fc06be471af3734b19b552aa2a6ade811e2))

- **15-02**: Create core/midline/ module with segment-then-extract backend and stage
  ([`07e9150`](https://github.com/tlancaster6/AquaPose/commit/07e91506e34557181b5de8ebdff2c06f088f1860))

- **15-03**: Create core/association/ module with RANSAC backend and AssociationStage
  ([`9d1937d`](https://github.com/tlancaster6/AquaPose/commit/9d1937d5f7d25b91176d17876ded7f22bf92178f))

- **15-04**: Create core/tracking/ module with Hungarian backend and stage
  ([`609b2cf`](https://github.com/tlancaster6/AquaPose/commit/609b2cffb4cf80175edbc83351a89185aec5d1df))

- **15-05**: Create ReconstructionStage with triangulation/curve_optimizer backends
  ([`392b6d2`](https://github.com/tlancaster6/AquaPose/commit/392b6d23d6e6a06fb57e598098375de04ec81b59))

- **16-01**: Add e2e regression test and update golden data generator to PosePipeline
  ([`e03d219`](https://github.com/tlancaster6/AquaPose/commit/e03d219fe73951784e7f500c4a44a1c4ccd446de))

- **16-01**: Create regression test infrastructure and per-stage tests
  ([`1e887e6`](https://github.com/tlancaster6/AquaPose/commit/1e887e69a9b3a263f78aec6d3d87f73b19956e8d))

- **16-02**: Clean up v1.0 pipeline.stages test imports and fix diagnose_tracking path
  ([`a63301a`](https://github.com/tlancaster6/AquaPose/commit/a63301a5f10d02cf6f13a65daf97397896ae1030))

- **17-01**: Implement TimingObserver for pipeline profiling
  ([`c833fae`](https://github.com/tlancaster6/AquaPose/commit/c833faedf4f87d17e130b4996feac6e830bf8551))

- **17-02**: Implement HDF5ExportObserver with frame-major layout
  ([`982e6f7`](https://github.com/tlancaster6/AquaPose/commit/982e6f7e5290249ddeb6afa661aca65d31fa43ff))

- **17-03**: Implement Overlay2DObserver for 2D reprojection video
  ([`5513d20`](https://github.com/tlancaster6/AquaPose/commit/5513d20c56fc4c6e24e9300fcfca92f19b1f7881))

- **17-04**: Implement Animation3DObserver with Plotly HTML viewer
  ([`1502b8c`](https://github.com/tlancaster6/AquaPose/commit/1502b8c8160c98c9fc6f2e5cc67e7bf296be2413))

- **17-05**: Implement DiagnosticObserver for stage output capture
  ([`9b98d3f`](https://github.com/tlancaster6/AquaPose/commit/9b98d3fff7b2db882edd90986c6f098aa9c2e224))

- **18-01**: Add CLI entrypoint and ConsoleObserver
  ([`c611205`](https://github.com/tlancaster6/AquaPose/commit/c611205f6d52139cfb9b6a15ea62d3e900f9ec78))

- **18-02**: Add diagnostic and benchmark mode tests
  ([`b770670`](https://github.com/tlancaster6/AquaPose/commit/b770670d8c983671590d076247a6c1e4dc28ece5))

- **18-03**: Add synthetic execution mode with SyntheticDataStage
  ([`62e94df`](https://github.com/tlancaster6/AquaPose/commit/62e94df96a528e46aaca3de1c503d405a20fb30e))

- **19-01**: Add AST-based import boundary and structural rule checker
  ([`6b0a2e5`](https://github.com/tlancaster6/AquaPose/commit/6b0a2e5ad7b43a1e3871023b872b1c3526083960))

- **19-01**: Wire import boundary checker as pre-commit hook
  ([`9bc8dd8`](https://github.com/tlancaster6/AquaPose/commit/9bc8dd84710be828e660ed1233b25ef443dbf9da))

- **19-02**: Add pytest smoke test wrapper and fix CLI invocation
  ([`359a396`](https://github.com/tlancaster6/AquaPose/commit/359a396024f265008cbaaffed76e85cb12ee7359))

- **20-01**: Move PipelineContext and Stage from engine/stages to core/context
  ([`d8f2e6a`](https://github.com/tlancaster6/AquaPose/commit/d8f2e6ab6d10599fd6d5c875a5d3dd8720f90a17))

- **20-03**: Extract observer assembly from CLI to engine/observer_factory.py
  ([`4634ab6`](https://github.com/tlancaster6/AquaPose/commit/4634ab6353257340c5b8d2e917e981b6716ba021))

- **20-05**: Extract shared camera-video discovery, split diagnostics.py
  ([`a3ebff5`](https://github.com/tlancaster6/AquaPose/commit/a3ebff5886d70670cfec998501b4f2378b2cd58b))

- **overlay**: Add scale parameter to downsize output video (default 0.5x)
  ([`8682425`](https://github.com/tlancaster6/AquaPose/commit/86824254d9e27520d97a1b32d384e0726b53bfe2))

- **quick-9**: Add init-config CLI subcommand
  ([`1d6eaf9`](https://github.com/tlancaster6/AquaPose/commit/1d6eaf9bd2a1a8104a5a79b27bc2884179068952))

### Refactoring

- **20-04**: Refactor TrackingStage to consume Stage 3 bundles as primary input
  ([`0fe1ebd`](https://github.com/tlancaster6/AquaPose/commit/0fe1ebda02e1324950ad53fb0e2265cb7cffe646))

### Testing

- **13-01**: Add 7 unit tests for Stage protocol and PipelineContext
  ([`3137e91`](https://github.com/tlancaster6/AquaPose/commit/3137e912e4d81a49bb63d065f5cebaca3f8ffddd))

- **13-03**: Add 9 unit tests for event dataclasses and EventBus/Observer
  ([`a17f3ed`](https://github.com/tlancaster6/AquaPose/commit/a17f3edd4ba426b334a91d537979074293bd249f))

- **13-engine-core**: Add 8 tests for PosePipeline orchestration
  ([`9736226`](https://github.com/tlancaster6/AquaPose/commit/9736226cedba954bce6241666bdb72c60830da84))

- **15-01**: Add interface tests for DetectionStage
  ([`c45f705`](https://github.com/tlancaster6/AquaPose/commit/c45f705aa508466e4dd21b4d6ab4b9b8faf45372))

- **15-02**: Add interface tests for MidlineStage
  ([`a3b74b8`](https://github.com/tlancaster6/AquaPose/commit/a3b74b8306214e5b14d8128840c9be189ed4daa6))

- **15-03**: Add interface tests for AssociationStage
  ([`551d5c9`](https://github.com/tlancaster6/AquaPose/commit/551d5c9412e611462c2874a97793e5f3c21a9b94))

- **15-04**: Add interface tests for TrackingStage
  ([`0cbeaf1`](https://github.com/tlancaster6/AquaPose/commit/0cbeaf160b049bd13d898dc8e800df7c576d2128))

- **15-05**: Interface tests for ReconstructionStage and full pipeline smoke test
  ([`6b37aae`](https://github.com/tlancaster6/AquaPose/commit/6b37aae58f75d898502a91c10db41bbb25a15c36))

- **20-04**: Update tracking stage tests for bundle-based input
  ([`586f4b1`](https://github.com/tlancaster6/AquaPose/commit/586f4b10d41424a8dad89d66cf7b64699200d686))


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
