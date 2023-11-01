# import redis

# redis_pool = redis.ConnectionPool(
#     host='192.168.0.48',
#     port=3379,
#     password='vitonguE@1@1',
#     db=0,
#     decode_responses=True)

# redis_conn = redis.Redis(connection_pool=redis_pool)
# message1 = {"chatId":"1c7ecc18-9f04-4261-a566-c47dfeda25f5","msgId":"a91b38487b174b9d8c3fc34e39f767a0","response":"123444 end!"}
# stream_name = f"momrah:sse:chat:a91b38487b174b9d8c3fc34e39f767a0"
# message1_id = redis_conn.xadd(stream_name, message1)

a = "What is your name?"
b = "What is your"
print(a.replace(b, ''))