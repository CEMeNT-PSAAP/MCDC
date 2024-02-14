#!/bin/bash

# Patch numba
s=$(python -c 'import numba; print(numba.__path__[0])')
s+='/core/types/scalars.py'

line=$(grep -n 'class Boolean(Hashable):' $s | cut -f1 -d:)

head -n $line $s > tmp-scalars.py
cat >> tmp-scalars.py << EOF
    def __init__(self, name):
        super(Boolean, self).__init__(name)
        self.bitwidth = 8
EOF

length=$(wc -l < $s)
let "length -= line"
tail -n $length $s >> tmp-scalars.py

mv tmp-scalars.py $s

