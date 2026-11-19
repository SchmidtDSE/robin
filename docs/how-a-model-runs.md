# How a model runs

This page tells what happens to a recording between the engine and the model's
outputs. It is for a reader who is new to the code. The settings that it names are
in [the profile](profile.md).

## The parts

- The **engine** runs one work. It fetches the model's files, builds the model once,
  and then runs the recordings one after another. For each recording, it fetches the
  audio, takes the model's windows one at a time, checks each window, keeps the scores
  that the request asks for, and writes them. Then it makes the detections and
  publishes the recording's files.
- An **adapter** knows one model. The engine calls only its `run`, which takes a
  recording and returns the recording's windows in order. The engine does not know how
  the adapter reads audio or calls its network.
- The **window loop** cuts a recording into windows, decodes their audio, and runs the
  network on them in batches. It is meant for every backbone model: a model that runs
  directly on audio. BirdNET, Perch, and OWL use it. The ONNX heads are not backbones,
  so they do not use it. See [OWL and the ONNX heads](#owl-and-the-onnx-heads).

## Why reading and scoring overlap

Each window needs two kinds of work. Its audio must be decoded and resampled, which
the CPU does. Then the network must run on it, on the CPU or the GPU. If the loop did
one and then the other, each would wait for the other. The GPU would have nothing to do
while the audio is decoded, and no audio would be decoded while the network runs.

So the window loop does both at the same time. Reader threads decode the next batches
while the model thread runs the network on the current one.

The readers also decode the audio in long blocks, not one window at a time. Each
separate read costs a seek and a new start of the decoder, which is slow for
compressed audio such as FLAC.

## A recording through the window loop

1. **Plan.** The loop counts the frames of the file and makes the list of windows from
   the window geometry: the window length, the overlap, and the padding. It puts the
   windows into batches of `batch_size` windows, and numbers the batches from 0.
2. **Read.** `read_threads` reader threads run. By default there is one fewer than the
   CPUs that the process can use, and at least 1. In a container with a CPU limit, the
   limit counts, not the host's CPUs. Each reader opens its own handle on the file. A reader takes the next batch
   number, and decodes the batch's windows in blocks of at most `read_block_seconds`.
   The adapter cuts each window from its block, at the model's sample rate. A batch is
   an array of `batch_size` rows. After the last window of the recording, the rows are
   zero, so every batch has the same shape.
3. **Limit.** A reader must take a free slot before it takes a batch. The model thread
   gives the slot back when it takes that batch. So no more than `read_ahead_batches`
   batches, plus one for each reader, are read and not yet taken. This limits the
   memory that a long recording uses.
4. **Score.** One model thread takes the batches in number order. If a reader finishes
   a batch before its turn, the batch waits. The model thread runs the network, drops
   the outputs of the zero rows, and makes one output for each window.
5. **Hand over.** The outputs go to the engine in window order, through a queue of no
   more than `read_ahead_batches` batches. The engine then checks and writes each
   window.

The readers can finish batches in any order, but the engine always gets the windows in
order.

## Inside `BatchPipeline`

`BatchPipeline` is the part of the window loop that runs the threads. The window loop
makes one `BatchPipeline` for each recording, after it has made the list of windows.

### What an adapter gives it

An adapter gives the window loop three functions. Only these three are different for
each model:

- a **block reader**, which decodes the audio of a block of windows and returns it in
  the adapter's own form;
- a **window cutter**, which cuts one window's samples from a block, at the model's
  sample rate;
- a **batch runner**, which runs the network on one batch and returns one output for
  each real window.

The adapter also gives the window length in samples and its resources: `batch_size`,
`read_threads`, `read_ahead_batches`, and `read_block_seconds`. `BatchPipeline` does the rest: it makes
the batches, pads them, runs the threads, and keeps the windows in order.

### The threads and the queues

No thread starts until the engine asks for the first window. Then `BatchPipeline`
starts the reader threads and one model thread. Each thread has a name that starts with
`robin-windows-`, so it is easy to find in a stack dump. Four things connect the
threads:

- **The batch numbers.** A queue holds the numbers 0, 1, 2, and so on. Each reader
  takes the next number, so each batch is read by one reader only.
- **The slots.** A count of free slots, `read_ahead_batches` plus one for each reader. A
  reader takes a slot before it takes a number, and the model thread gives the slot back
  when it takes that batch. When no slot is free, the readers wait. This is what limits
  the memory.
- **The read batches.** A queue where each reader puts each batch that it has read,
  with the batch's number. The model thread takes from it. A batch that arrives before
  its turn is kept until the batches before it have run.
- **The outputs.** A queue of no more than `read_ahead_batches` items, from the model
  thread to the engine. Each item is the outputs of one batch, an error, or a mark that
  says the last batch is done.

The model thread makes the whole list of a batch's outputs before it puts the list in
the queue. So if the adapter fails while it makes the outputs, no window of that batch
reaches the engine.

As soon as the model thread has run a batch, it lets go of the batch's audio, so the
memory is free before it waits for the next batch. A reader does the same after it puts
a batch in the queue.

### Errors and stopping

- An error in a reader or in the model thread goes through the queues in place of a
  batch. So it reaches the engine after the windows of all the batches before it. If a
  reader cannot open the file, the error takes the place of the batch that the reader
  took.
- When the engine stops taking windows, for example after an error of its own,
  `BatchPipeline` sets a stop flag. A thread that waits checks the flag every 0.1
  seconds, and a reader also checks it between blocks. Each thread then stops, and each
  reader closes its own file. `BatchPipeline` waits for every thread to end, and then
  empties the queues, so that no audio stays in memory after the recording.

### What it counts

`BatchPipeline` adds up three times: the seconds that each reader spends reading, the
seconds in the batch runner, and the seconds that the model thread waits for a batch.
The window loop writes them in the log line below.

## The settings change speed, not results

`batch_size`, `read_threads`, `read_ahead_batches`, and `read_block_seconds` change only
speed and memory. Each window gets the same samples and the same scores with any value:

- A block read gives each window the same samples that a read of that window alone
  would give.
- BirdNET resamples each window on its own, as the BirdNET library does.
- Perch resamples each block with up to 1 second of extra audio on each side. The
  resampling filter reaches much less than 1 second, so each sample is the same as when
  the whole recording is resampled at once.
- The zero rows of a batch change no output of a real window.

The tests check each of these.

## Reading the log

When a recording's last window reaches the engine, the window loop writes one line. For
example:

```text
ran 720 windows of a.flac: read 41.3 s, model 18.0 s, waited for audio 0.4 s
```

- `read` is the time that the readers spent decoding, added over all readers. So it can
  be more than the time that the recording took.
- `model` is the time spent in the network calls.
- `waited for audio` is the time that the model thread spent waiting for a batch.

If `waited for audio` is near 0, the model is the slow part, and the readers keep up.
If it is large, reading is the slow part. More readers can help, if the machine has
CPUs to spare: set `read_threads`. Longer blocks can also help. The loop's first line for
a recording gives the number of read threads.

The engine then writes its own line for the recording, with the seconds of its stages:
fetch, engine, detections, and publish. `fetch` is the time that the recording waited
for its input, and for the checksum of its audio. The engine fetches each input while
the recording before it runs, so `fetch` is near 0 when the fetch finished in that time.
The first recording waits for the whole of its fetch.

## What still runs one after another

The engine runs the recordings one at a time. While a recording runs, one background
thread fetches the next recording's input. The engine fetches only one recording ahead,
because a long recording can be large on disk. If the next recording never runs, because
the work stops, the engine releases its input. If a fetch fails, only the recording that
it was for fails. The model is built once for each work, and it stays loaded for all the
recordings.

The same thread computes the `sha256:` checksum of each audio file that it fetches. The
result names it as the recording's `input_checksum` in `coverage`, so that a later change
to the audio can be found. Saved embeddings get none: the work pins their checksum
already. A file that cannot be read for its checksum fails its recording with
`input_unavailable`.

## OWL and the ONNX heads

- **OWL** uses the window loop, but its network takes a spectrogram, not samples. So
  its window cutter does more than the others:
  - It makes the window's spectrogram from the window's samples. It writes no image
    file.
  - It makes the spectrogram with sox_numpy, which uses NumPy and no TensorFlow. So the
    spectrogram is made on the CPU, also on a GPU worker, and takes no GPU time from the
    network. The reader threads make the spectrograms of the next batches while the
    network runs on the current one.
  - sox_numpy resamples with soxr, whose OpenMP threads use much more CPU than they
    save. The reader threads are limited by the CPU, so the adapter sets
    `OMP_NUM_THREADS=1` before it loads sox_numpy, unless the caller has set it.
  - It changes each pixel to the gray value that the network expects, and puts the
    spectrogram into one batch row.

  OWL's batch runner does not give the zero rows of the last batch to the network. The
  network gets only the real windows. The rows that are in a batch change the scores a
  little, at the level of float32 rounding. Without the zero rows, OWL's scores on the
  CPU are exactly the same as before OWL used the window loop.
- **An ONNX head** reads no audio. It runs on the embeddings that a backbone model has
  already made, in batches, and only on the CPU.

## Where the code is

- `inference_engine/robin_inference_engine/engine.py` runs the work and each
  recording's stages.
- `models/robin_models/_adapter_support/` holds what the adapters share:
  - `audio_windows.py` starts the window loop for a recording, and holds the helpers
    that make the outputs of a batch.
  - `batch_pipeline.py` holds `BatchPipeline`, with the reader threads and the model
    thread.
  - `adapter_setup.py` reads the resources, and checks the card and the files.
  - `saved_model.py` loads and checks a TensorFlow SavedModel.
- `models/robin_models/birdnet/`, `models/robin_models/perch/`, and
  `models/robin_models/owl/` each give the window loop its three functions: the block
  reader, the window cutter, and the batch runner.
