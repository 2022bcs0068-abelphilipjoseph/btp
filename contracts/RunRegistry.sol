// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract RunRegistry {
    struct Run {
        bytes32 artifactHash;   // sha256 hash of artifact or metadata
        string artifactType;    // e.g. "metrics-json", "model-checkpoint"
        string metadataURI;     // e.g. ipfs://CID
        uint256 timestamp;
        address owner;
    }

    Run[] public runs;
    event RunRegistered(uint256 indexed runId, address indexed owner, bytes32 artifactHash, string artifactType, string metadataURI, uint256 timestamp);

    function registerRun(bytes32 artifactHash, string memory artifactType, string memory metadataURI) external returns (uint256) {
        runs.push(Run({
            artifactHash: artifactHash,
            artifactType: artifactType,
            metadataURI: metadataURI,
            timestamp: block.timestamp,
            owner: msg.sender
        }));
        uint256 id = runs.length - 1;
        emit RunRegistered(id, msg.sender, artifactHash, artifactType, metadataURI, block.timestamp);
        return id;
    }

    function getRunCount() external view returns (uint256) {
        return runs.length;
    }

    function getRun(uint256 idx) external view returns (bytes32, string memory, string memory, uint256, address) {
        require(idx < runs.length, "idx OOB");
        Run memory r = runs[idx];
        return (r.artifactHash, r.artifactType, r.metadataURI, r.timestamp, r.owner);
    }
}
