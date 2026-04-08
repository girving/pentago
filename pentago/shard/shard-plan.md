# Shard generation plan

Generate ~100 GB of shard files (8500 of 1048576 total shards, slices 0–18) for initial
ML training.

## Prep

```bash
# Create the output bucket
gcloud storage buckets create gs://pentago-shards --location=us-central1

# Move slices 17–18 out of coldline to avoid retrieval fees on future reads.
# Each command does a server-side rewrite that requires multiple round-trips,
# so run from Cloud Shell or a GCE instance to avoid losing progress if your
# local machine disconnects. Safe to re-run if interrupted (original object
# stays intact until the rewrite completes).
gcloud storage objects update gs://pentago-us-central1/slice-17.pentago --storage-class=standard
gcloud storage objects update gs://pentago-us-central1/slice-18.pentago --storage-class=standard
```

The rewrite takes a while (~3.4 TB to read and rewrite). Can run overnight.

## Service account

```bash
# Create a service account for the sharder
gcloud iam service-accounts create pentago-sharder --display-name="Pentago sharder"
SA=pentago-sharder@$(gcloud config get-value project).iam.gserviceaccount.com

# Grant read access to input bucket and write access to output bucket
gcloud storage buckets add-iam-policy-binding gs://pentago-us-central1 \
  --member="serviceAccount:$SA" --role=roles/storage.objectViewer
gcloud storage buckets add-iam-policy-binding gs://pentago-shards \
  --member="serviceAccount:$SA" --role=roles/storage.objectAdmin

# Download key file
gcloud iam service-accounts keys create ~/pentago-sharder-key.json --iam-account=$SA
```

## Instance setup

```bash
# 96 vCPU, 372 GB RAM. Enough RAM for all 8500 shards in one batch (~230 GB).
# No local SSD needed — data streams directly from/to GCS.
gcloud compute instances create pentago-sharder \
  --zone=us-central1-a \
  --machine-type=c4-standard-96 \
  --boot-disk-size=50GB \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud

gcloud compute ssh pentago-sharder --zone=us-central1-a
```

```bash
sudo apt-get update && sudo apt-get install -y git build-essential
sudo curl -Lo /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazelisk
sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel
```

## Copy credentials

```bash
# From your local machine:
gcloud compute scp ~/pentago-sharder-key.json pentago-sharder:~/key.json --zone=us-central1-a
```

## Build and run

```bash
cd ~
git clone https://github.com/girving/pentago.git
cd pentago
bin/bazel build -c opt //pentago/shard:sharder

bazel-bin/pentago/shard/sharder \
  --max-slice 18 \
  --shards 1048576 \
  --range :8500 \
  --credentials ~/key.json \
  gs://pentago-us-central1 \
  gs://pentago-shards \
; sudo shutdown -h now
```

The `shutdown -h now` stops (not terminates) the instance when the sharder
finishes, regardless of success or failure. Restart with
`gcloud compute instances start pentago-sharder` if needed.
Stopped instances cost only ~$0.04/month for the boot disk.

The sharder reads supertensor files directly from GCS (chunk-cached, ~64 MB chunks)
and writes shard files directly to GCS. No local disk staging needed.
Blocks are sorted by file offset for sequential I/O, so the chunk cache streams
through each supertensor file approximately once.

With 96 vCPU doing parallel lzma decompression, the sharder should process
all 4.3 TB in ~30–40 minutes (~$3 total at ~$4/hr).
Progress is logged per-block per-slice.

## Verify output

```bash
# Check shard count
gcloud storage ls gs://pentago-shards/ | wc -l  # should be 8500

# Spot-check a shard
gcloud storage ls -l gs://pentago-shards/shard-0000000-of-1048575.pentago.shard
```

## Tear down

```bash
gcloud compute instances delete pentago-sharder --zone=us-central1-a
```
## Sizing notes

| Parameter | Value |
|-----------|-------|
| Total shards | 1,048,576 |
| Shard range | :8500 (first 8500 shards) |
| Output size | ~100 GB |
| Per-shard size | ~12 MB |
| Filename pattern | `shard-0000000-of-1048575.pentago.shard` |

## Revert slices 17–18 to coldline

Once all shard generation is complete (including future full runs), revert to coldline.
No retrieval fee in this direction.

```bash
gcloud storage objects update gs://pentago-us-central1/slice-17.pentago --storage-class=coldline
gcloud storage objects update gs://pentago-us-central1/slice-18.pentago --storage-class=coldline
```

## Future: full shard generation

To generate all 1M shards later, use `--range 8500:` on a similar instance.
The sharder supports split ranges, so previously generated shards don't need
to be regenerated. Total output: ~12.5 TB across 1,048,576 files.
