import numpy as np
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType

# Load your data
X = np.load("data/X.npy").astype("float32")  # make sure it's float32 for Milvus
y = np.load("data/y.npy", allow_pickle=True)

# Vector dimensions
dim = X.shape[1]
collection_name = "jet_vectors"

# Connect to Zilliz/Milvus
connections.disconnect("default")
connections.connect(
    alias="default",
    uri="https://in03-617b3bd148a1a68.serverless.gcp-us-west1.cloud.zilliz.com",  # Replace with your cluster
    user="db_617b3bd148a1a68",
    password="Or3{mZ7&~mSufLQl",
    secure=True
)

# Drop if exists
if utility.has_collection(collection_name):
    Collection(collection_name).drop()

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=20)
]
schema = CollectionSchema(fields)
collection = Collection(name=collection_name, schema=schema)
collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
collection.load()

# Batch insert
BATCH_SIZE = 500
def insert_batches(X, y):
    for i in range(0, len(X), BATCH_SIZE):
        xb = X[i:i+BATCH_SIZE]
        yb = y[i:i+BATCH_SIZE]
        collection.insert([xb.tolist(), yb.tolist()])
        print(f"Inserted {i + len(xb)} / {len(X)}")

insert_batches(X, y)
print("✅ Done inserting jet data.")
