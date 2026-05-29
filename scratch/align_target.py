import json

# Load metadata
meta = json.load(open('outputs/alphafold_db/AF-Q9E006-F1-metadata.json'))
full_seq = meta.get('uniprotSequence', '')

target = "IYELKMECPHTVGLGQGYIIGSTELGLISIEAASDIKLESSCNFDLHTTSMAQKSFTQVEWRKKSDTTDTTNAASTTFEAQTKTVNLRGTCILAPELYDTVKKTVLCYDLTCNQTHCQPTVYLIAPVLTCMSIRSCMASVFTSRIQVIYEKTHCVTGQLIEGQCFNPAHTLTLSQPAHTYDTVTLPISCFFTPKKSEQLKVIKTFEGILTKTGCTENALQGYYVCFLGSHSEPLIVPSLEDIRSAEVVSRMLVHPRGEDHDAIQNSQSHLRIVGPITAKVPSTSSTDTLKGTAFAGVPMYSSLSTLVRNADPEFVFSPGIVPESNHSTCDKKTVPITWTGYLPISGEME"

print(f"Target length: {len(target)}")
print(f"Polyprotein length: {len(full_seq)}")

# Sliding window alignment (find best match of size 30 for each region)
step = 10
window_size = 20
for i in range(0, len(target) - window_size + 1, step):
    sub = target[i:i+window_size]
    idx = full_seq.find(sub)
    if idx != -1:
        print(f"Target [{i}:{i+window_size}] -> Polyprotein [{idx}:{idx+window_size}]")
    else:
        # Find closest match
        best_score = 0
        best_idx = -1
        for j in range(len(full_seq) - window_size + 1):
            score = sum(1 for k in range(window_size) if sub[k] == full_seq[j+k])
            if score > best_score:
                best_score = score
                best_idx = j
        print(f"Target [{i}:{i+window_size}] -> Polyprotein [{best_idx}:{best_idx+window_size}] (score {best_score}/{window_size})")
