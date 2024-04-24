Mostly refined the code in agent trainer and added some new methods in Chunk and PowerReplay

Added the functinality of adding a chunk of larger size to the power_replay by agent trainer

NOTE : In the new implementation:
- It is agent trainer's responsibility to first create the chunk and then pass it to the power replay where the chunk will then be added to the buffer
- So now agent trainer can not just call power_replay to add one transition or bunch of transition. Agent trainer has to pass a chunk to the power replay.
