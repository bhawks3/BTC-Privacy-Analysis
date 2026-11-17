## Iteration 1

SELECT COUNT(DISTINCT to_out.txid)  
FROM tx_out to_out  
JOIN transactions t ON to_out.txid = t.txid  
JOIN blocks b ON t.hashBlock = b.block_hash  
WHERE b.height BETWEEN {start_block} AND {end_block}  
AND (  
(to_out.scriptPubKey LIKE ’%OP_RETURN%’ AND LENGTH(  
to_out.scriptPubKey) IN (70, 138))  
OR to_out.scriptPubKey LIKE ’%02020202%’  
OR to_out.scriptPubKey LIKE ’%03030303%’  
);

## Iteration 2
WHERE ... AND (  
(to_out.scriptPubKey LIKE ’%OP_RETURN%’ AND LENGTH(to_out  
.scriptPubKey) BETWEEN 50 AND 150)  
OR to_out.scriptPubKey LIKE ’%02%’  
OR to_out.scriptPubKey LIKE ’%03%’  
OR to_out.scriptPubKey LIKE ’%OP_RETURN%’  
OR ( NOT ( ...standard script patterns... ) AND LENGTH(  
to_out.scriptPubKey) > 40 )  
)

## Iteration 3
WHERE ... AND (   
(to_out.scriptPubKey LIKE ’6a26%’ AND LENGTH(to_out.  
scriptPubKey) = 70)   
OR  
(to_out.scriptPubKey LIKE ’21%’ AND (to_out.scriptPubKey  
LIKE ’2102%’ OR to_out.scriptPubKey LIKE ’2103%’)  
AND LENGTH(to_out.scriptPubKey) = 68)  
)
