import pickle

import joblib
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

enc = tiktoken.get_encoding("cl100k_base")


# # 字节对编码过程，我的输出是[31373, 995]
# encoding_res = enc.encode("$ModuleBasePath = Split-Path $MyInvocation.MyCommand.Path -Parent")
# print(encoding_res)
#
# # 字节对解码过程，解码结果：hello world
# raw_text = enc.decode(encoding_res)
# print(raw_text)
# print([enc.decode_single_token_bytes(token) for token in encoding_res])

def my_tokenizer(sentence: str):
    encoding_res = enc.encode(sentence)
    return [enc.decode_single_token_bytes(token).decode().strip(" ") for token in encoding_res]


train_scripts_ = [
    "Write-Host (Get-WmiObject -Namespace root\wmi -Class MSiSCSIInitiator_MethodClass).iSCSINodeName",
    "Get-WmiObject -Namespace root\wmi -Class MSiSCSIInitiator_SendTargetPortalClass | Foreach-Object { Write-Host $_.PortalAddress }",
    "Write-Host (Get-WmiObject -Namespace root\wmi -Class MSiSCSIInitiator_MethodClass).iSCSINodeName",
    "Get-WmiObject -Namespace root\wmi -Class MSiSCSIInitiator_SendTargetPortalClass | Foreach-Object { Write-Host $_.PortalAddress }",

]
vectorizer_ = TfidfVectorizer(max_features=30, ngram_range=(1, 1), tokenizer=my_tokenizer)
train_features_ = vectorizer_.fit_transform(train_scripts_)
words = vectorizer_.get_feature_names_out()
print(words)

vectorizer_ = TfidfVectorizer(max_features=30, ngram_range=(1, 1))
train_features_ = vectorizer_.fit_transform(train_scripts_)
words = vectorizer_.get_feature_names_out()
print(words)
