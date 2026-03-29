import cereal.messaging as messaging

def publish_debug(pm, pred):
    msg = messaging.new_message("alpamayoDebug")
    msg.alpamayoDebug.valid = bool(pred.get("valid", True))
    msg.alpamayoDebug.text = str(pred.get("text", ""))
    msg.alpamayoDebug.confidence = float(pred.get("confidence", 0.0))
    msg.alpamayoDebug.mode = str(pred.get("mode", "unknown"))
    pm.send("alpamayoDebug", msg)