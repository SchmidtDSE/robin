# Schema migrations

Each `.sql` file in this folder is one migration, named `NNNN_name.sql` and numbered
from 0001 with no gap. `robin db migrate` applies them in order. An applied file must
never change; a change to the schema is a new file.
