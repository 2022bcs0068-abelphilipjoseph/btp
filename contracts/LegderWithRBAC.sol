// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LedgerWithRBAC {
    
    // --- STATE VARIABLES ---
    address public admin;
    mapping(address => bool) public authorizedAuditors;

    // The structure of our AI Fairness Report
    struct Audit {
        string modelVersion;
        string reportUrl;
        uint256 cosineBiasScore;
        uint256 weatScore;
        uint256 counterFactualDist;
        address auditor;
        uint256 timestamp;
    }

    // The immutable ledger (array) of all verified audits
    Audit[] public audits;

    // --- EVENTS ---
    // Emits a public log every time a successful audit is recorded
    event AuditLogged(string modelVersion, address indexed auditor, uint256 timestamp);

    // --- CONSTRUCTOR ---
    constructor() {
        admin = msg.sender; 
        authorizedAuditors[msg.sender] = true;
    }

    // --- MODIFIERS ---
    
    modifier onlyAuditor() {
        require(authorizedAuditors[msg.sender] == true, "ACCESS DENIED: You are not a verified AI Auditor.");
        _;
    }

    // --- ADMIN FUNCTIONS ---
    function authorizeAuditor(address _auditorWallet) public {
        require(msg.sender == admin, "ACCESS DENIED: Only the Admin can authorize new auditors.");
        authorizedAuditors[_auditorWallet] = true;
    }

    function revokeAuditor(address _auditorWallet) public {
        require(msg.sender == admin, "ACCESS DENIED: Only the Admin can revoke auditors.");
        authorizedAuditors[_auditorWallet] = false;
    }

    // --- CORE LOGIC (WRITE) ---
    // Locked down by the 'onlyAuditor' modifier
    function logAudit(
        string memory _modelVersion, 
        string memory _reportUrl, 
        uint256 _cosineBiasScore, 
        uint256 _weatScore, 
        uint256 _counterFactualDist
    ) public onlyAuditor {
        
        // Push the new audit into the blockchain array
        audits.push(Audit({
            modelVersion: _modelVersion,
            reportUrl: _reportUrl,
            cosineBiasScore: _cosineBiasScore,
            weatScore: _weatScore,
            counterFactualDist: _counterFactualDist,
            auditor: msg.sender, // Cryptographically tied to the signer
            timestamp: block.timestamp
        }));

        emit AuditLogged(_modelVersion, msg.sender, block.timestamp);
    }

    // --- READ FUNCTIONS ---
    function getAuditCount() public view returns (uint256) {
        return audits.length;
    }

    function getAudit(uint256 _index) public view returns (
        string memory, string memory, uint256, uint256, uint256, address, uint256
    ) {
        require(_index < audits.length, "Audit does not exist.");
        Audit memory a = audits[_index];
        return (a.modelVersion, a.reportUrl, a.cosineBiasScore, a.weatScore, a.counterFactualDist, a.auditor, a.timestamp);
    }
}