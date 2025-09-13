"""
Regulatory Compliance Manager
Full FCA/CFTC compliance features with audit trails

This module provides:
- FCA (Financial Conduct Authority) compliance
- CFTC (Commodity Futures Trading Commission) compliance
- Automated compliance monitoring
- Audit trail management
- Regulatory reporting
- Compliance alerts and notifications
- Risk limit enforcement
- Transaction reporting

Month 5-6: Advanced Features and Automation
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

def serialize_datetime(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj


class RegulatoryAuthority(str, Enum):
    """Regulatory authorities"""
    FCA="FCA"  # UK Financial Conduct Authority
    CFTC = "CFTC"  # US Commodity Futures Trading Commission
    SEC = "SEC"  # US Securities and Exchange Commission
    ESMA = "ESMA"  # European Securities and Markets Authority
    MAS = "MAS"  # Monetary Authority of Singapore


class ComplianceRule(str, Enum):
    """Compliance rules"""
    POSITION_LIMITS="position_limits"
    RISK_LIMITS = "risk_limits"
    REPORTING_REQUIREMENTS = "reporting_requirements"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    MARKET_ABUSE = "market_abuse"
    BEST_EXECUTION = "best_execution"
    CLIENT_MONEY = "client_money"
    TRANSPARENCY = "transparency"


class ComplianceStatus(str, Enum):
    """Compliance status"""
    COMPLIANT="compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    CRITICAL = "critical"
    UNDER_REVIEW = "under_review"


@dataclass
class ComplianceRuleDefinition:
    """Compliance rule definition"""
    rule_id: str
    authority: RegulatoryAuthority
    rule_type: 'ComplianceRule'
    description: str
    threshold: float
    measurement_period: int  # days
    severity: str  # "low", "medium", "high", "critical"
    is_active: bool=True
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime=field(default_factory=datetime.now)


@dataclass
class ComplianceCheck:
    """Compliance check result"""
    check_id: str
    rule_id: str
    timestamp: datetime
    status: ComplianceStatus
    current_value: float
    threshold_value: float
    deviation: float
    severity: str
    details: Dict[str, Any]
    remediation_actions: List[str]
    reviewed_by: Optional[str] = None
    review_date: Optional[datetime] = None


@dataclass
class AuditTrail:
    """Audit trail entry"""
    entry_id: str
    timestamp: datetime
    user_id: str
    action: str
    entity_type: str
    entity_id: str
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    reason: str
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


class RegulatoryComplianceManager:
    """
    Regulatory compliance management system
    
    Provides:
    - Automated compliance monitoring
    - Audit trail management
    - Regulatory reporting
    - Compliance alerts
    - Risk limit enforcement
    """
    
    def __init__(self, 
                 primary_authority: RegulatoryAuthority=RegulatoryAuthority.FCA,
                 enable_audit_trail: bool=True,
                 compliance_db_path: str="compliance.db"):
        """
        Initialize regulatory compliance manager
        
        Args:
            primary_authority: Primary regulatory authority
            enable_audit_trail: Enable audit trail logging
            compliance_db_path: Path to compliance database
        """
        self.primary_authority=primary_authority
        self.enable_audit_trail = enable_audit_trail
        self.compliance_db_path = compliance_db_path
        
        self.logger = logging.getLogger(__name__)
        
        # Compliance state
        self.compliance_rules: Dict[str, ComplianceRuleDefinition] = {}
        self.compliance_checks: List[ComplianceCheck] = []
        self.audit_trail: List[AuditTrail] = []
        
        # Performance tracking
        self.check_count=0
        self.violation_count = 0
        self.last_compliance_check = None
        
        # Initialize database
        if self.enable_audit_trail:
            self._init_compliance_database()
        
        # Load default compliance rules
        self._load_default_compliance_rules()
        
        self.logger.info(f"Regulatory Compliance Manager initialized for {primary_authority}")
    
    def _init_compliance_database(self):
        """Initialize compliance database"""
        try:
            with sqlite3.connect(self.compliance_db_path) as conn:
                cursor=conn.cursor()
                
                # Compliance rules table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_rules (
                        rule_id TEXT PRIMARY KEY,
                        authority TEXT NOT NULL,
                        rule_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        measurement_period INTEGER NOT NULL,
                        severity TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Compliance checks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS compliance_checks (
                        check_id TEXT PRIMARY KEY,
                        rule_id TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        deviation REAL NOT NULL,
                        severity TEXT NOT NULL,
                        details TEXT,
                        remediation_actions TEXT,
                        reviewed_by TEXT,
                        review_date TIMESTAMP,
                        FOREIGN KEY (rule_id) REFERENCES compliance_rules (rule_id)
                    )
                """)
                
                # Audit trail table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        entry_id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        old_values TEXT,
                        new_values TEXT,
                        reason TEXT NOT NULL,
                        ip_address TEXT,
                        session_id TEXT
                    )
                """)
                
                # Regulatory reports table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS regulatory_reports (
                        report_id TEXT PRIMARY KEY,
                        authority TEXT NOT NULL,
                        report_type TEXT NOT NULL,
                        period_start TIMESTAMP NOT NULL,
                        period_end TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        data TEXT NOT NULL,
                        submitted_date TIMESTAMP,
                        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
            self.logger.info("Compliance database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing compliance database: {e}")
    
    def _load_default_compliance_rules(self):
        """Load default compliance rules"""
        try:
            # FCA compliance rules
            if self.primary_authority== RegulatoryAuthority.FCA:
                self._load_fca_rules()
            
            # CFTC compliance rules
            elif self.primary_authority== RegulatoryAuthority.CFTC:
                self._load_cftc_rules()
            
            # Generic compliance rules
            else:
                self._load_generic_rules()
            
            self.logger.info(f"Loaded {len(self.compliance_rules)} compliance rules")
            
        except Exception as e:
            self.logger.error(f"Error loading compliance rules: {e}")
    
    def _load_fca_rules(self):
        """Load FCA-specific compliance rules"""
        fca_rules=[
            ComplianceRuleDefinition(
                rule_id="FCA_POSITION_LIMIT_001",
                authority=RegulatoryAuthority.FCA,
                rule_type=ComplianceRule.POSITION_LIMITS,
                description="Maximum position size as percentage of portfolio",
                threshold=0.20,  # 20%
                measurement_period=1,
                severity="high"
            ),
            ComplianceRuleDefinition(
                rule_id="FCA_RISK_LIMIT_001",
                authority=RegulatoryAuthority.FCA,
                rule_type=ComplianceRule.RISK_LIMITS,
                description="Maximum daily VaR as percentage of portfolio",
                threshold=0.05,  # 5%
                measurement_period=1,
                severity="critical"
            ),
            ComplianceRuleDefinition(
                rule_id="FCA_CAPITAL_001",
                authority=RegulatoryAuthority.FCA,
                rule_type=ComplianceRule.CAPITAL_REQUIREMENTS,
                description="Minimum capital adequacy ratio",
                threshold=0.08,  # 8%
                measurement_period=30,
                severity="critical"
            ),
            ComplianceRuleDefinition(
                rule_id="FCA_REPORTING_001",
                authority=RegulatoryAuthority.FCA,
                rule_type=ComplianceRule.REPORTING_REQUIREMENTS,
                description="Daily risk reporting threshold",
                threshold=0.03,  # 3%
                measurement_period=1,
                severity="medium"
            )
        ]
        
        for rule in fca_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    def _load_cftc_rules(self):
        """Load CFTC-specific compliance rules"""
        cftc_rules=[
            ComplianceRuleDefinition(
                rule_id="CFTC_POSITION_LIMIT_001",
                authority=RegulatoryAuthority.CFTC,
                rule_type=ComplianceRule.POSITION_LIMITS,
                description="Maximum speculative position limits",
                threshold=0.15,  # 15%
                measurement_period=1,
                severity="high"
            ),
            ComplianceRuleDefinition(
                rule_id="CFTC_RISK_LIMIT_001",
                authority=RegulatoryAuthority.CFTC,
                rule_type=ComplianceRule.RISK_LIMITS,
                description="Maximum daily risk exposure",
                threshold=0.04,  # 4%
                measurement_period=1,
                severity="critical"
            ),
            ComplianceRuleDefinition(
                rule_id="CFTC_REPORTING_001",
                authority=RegulatoryAuthority.CFTC,
                rule_type=ComplianceRule.REPORTING_REQUIREMENTS,
                description="Large trader reporting threshold",
                threshold=0.02,  # 2%
                measurement_period=1,
                severity="medium"
            )
        ]
        
        for rule in cftc_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    def _load_generic_rules(self):
        """Load generic compliance rules"""
        generic_rules=[
            ComplianceRuleDefinition(
                rule_id="GEN_POSITION_LIMIT_001",
                authority=self.primary_authority,
                rule_type=ComplianceRule.POSITION_LIMITS,
                description="Maximum position size limit",
                threshold=0.25,  # 25%
                measurement_period=1,
                severity="high"
            ),
            ComplianceRuleDefinition(
                rule_id="GEN_RISK_LIMIT_001",
                authority=self.primary_authority,
                rule_type=ComplianceRule.RISK_LIMITS,
                description="Maximum risk exposure limit",
                threshold=0.06,  # 6%
                measurement_period=1,
                severity="critical"
            )
        ]
        
        for rule in generic_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    async def run_compliance_checks(self, 
                                  portfolio_data: Dict[str, Any],
                                  risk_metrics: Dict[str, Any]) -> List[ComplianceCheck]:
        """
        Run all compliance checks
        
        Args:
            portfolio_data: Current portfolio data
            risk_metrics: Current risk metrics
            
        Returns:
            List[ComplianceCheck]: Compliance check results
        """
        try:
            checks=[]
            
            for rule_id, rule in self.compliance_rules.items():
                if rule.is_active:
                    check=await self._run_compliance_check(rule, portfolio_data, risk_metrics)
                    if check:
                        checks.append(check)
                        self.compliance_checks.append(check)
            
            # Update tracking
            self.check_count += len(checks)
            self.violation_count += len([c for c in checks if c.status != ComplianceStatus.COMPLIANT])
            self.last_compliance_check=datetime.now()
            
            # Log audit trail
            if self.enable_audit_trail:
                await self._log_audit_trail(
                    user_id="system",
                    action="compliance_check",
                    entity_type="portfolio",
                    entity_id="main",
                    old_values={},
                    new_values={"checks_run":len(checks), "violations":self.violation_count},
                    reason="Scheduled compliance check"
                )
            
            self.logger.info(f"Ran {len(checks)} compliance checks, {self.violation_count} violations")
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error running compliance checks: {e}")
            return []
    
    async def _run_compliance_check(self, 
                                  rule: ComplianceRuleDefinition,
                                  portfolio_data: Dict[str, Any],
                                  risk_metrics: Dict[str, Any]) -> Optional[ComplianceCheck]:
        """Run individual compliance check"""
        try:
            # Get current value based on rule type
            current_value=await self._get_current_value(rule, portfolio_data, risk_metrics)
            
            if current_value is None:
                return None
            
            # Check compliance
            if rule.rule_type== ComplianceRule.POSITION_LIMITS:
                is_compliant = current_value <= rule.threshold
            elif rule.rule_type == ComplianceRule.RISK_LIMITS:
                is_compliant = current_value <= rule.threshold
            elif rule.rule_type == ComplianceRule.CAPITAL_REQUIREMENTS:
                is_compliant = current_value >= rule.threshold
            else:
                is_compliant = current_value <= rule.threshold
            
            # Determine status
            if is_compliant:
                status = ComplianceStatus.COMPLIANT
            else:
                if rule.severity == "critical":status = ComplianceStatus.CRITICAL
                elif rule.severity == "high":status = ComplianceStatus.NON_COMPLIANT
                else:
                    status = ComplianceStatus.WARNING
            
            # Calculate deviation
            deviation = current_value - rule.threshold
            
            # Generate remediation actions
            remediation_actions = await self._generate_remediation_actions(rule, current_value, deviation)
            
            # Create check result
            check=ComplianceCheck(
                check_id=f"{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                timestamp=datetime.now(),
                status=status,
                current_value=current_value,
                threshold_value=rule.threshold,
                deviation=deviation,
                severity=rule.severity,
                details={
                    "rule_description":rule.description,
                    "measurement_period":rule.measurement_period,
                    "authority":rule.authority.value
                },
                remediation_actions=remediation_actions
            )
            
            return check
            
        except Exception as e:
            self.logger.error(f"Error running compliance check {rule.rule_id}: {e}")
            return None
    
    async def _get_current_value(self, 
                               rule: ComplianceRuleDefinition,
                               portfolio_data: Dict[str, Any],
                               risk_metrics: Dict[str, Any]) -> Optional[float]:
        """Get current value for compliance check"""
        try:
            if rule.rule_type== ComplianceRule.POSITION_LIMITS:
                # Get maximum position size
                positions = portfolio_data.get('positions', {})
                if not positions:
                    return 0.0
                
                total_value=sum(pos.get('value', 0) for pos in positions.values())
                if total_value== 0:
                    return 0.0
                
                max_position = max(pos.get('value', 0) for pos in positions.values())
                return max_position / total_value
            
            elif rule.rule_type== ComplianceRule.RISK_LIMITS:
                # Get VaR
                return risk_metrics.get('portfolio_var', 0.0)
            
            elif rule.rule_type== ComplianceRule.CAPITAL_REQUIREMENTS:
                # Get capital ratio
                return portfolio_data.get('capital_ratio', 0.0)
            
            elif rule.rule_type== ComplianceRule.REPORTING_REQUIREMENTS:
                # Get risk level for reporting
                return risk_metrics.get('portfolio_var', 0.0)
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error getting current value for {rule.rule_id}: {e}")
            return None
    
    async def _generate_remediation_actions(self, 
                                          rule: ComplianceRuleDefinition,
                                          current_value: float,
                                          deviation: float) -> List[str]:
        """Generate remediation actions for compliance violations"""
        try:
            actions=[]
            
            if rule.rule_type == ComplianceRule.POSITION_LIMITS:
                if deviation > 0:
                    actions.append(f"Reduce position size by {deviation:.2%}")
                    actions.append("Consider portfolio rebalancing")
                    actions.append("Review position sizing methodology")
            
            elif rule.rule_type== ComplianceRule.RISK_LIMITS:
                if deviation > 0:
                    actions.append(f"Reduce portfolio risk by {deviation:.2%}")
                    actions.append("Implement additional hedging")
                    actions.append("Review risk management policies")
            
            elif rule.rule_type== ComplianceRule.CAPITAL_REQUIREMENTS:
                if deviation < 0:
                    actions.append(f"Increase capital by {abs(deviation):.2%}")
                    actions.append("Review capital allocation")
                    actions.append("Consider capital injection")
            
            # Generic actions
            if rule.severity== "critical":
                actions.append("Immediate management notification required")
                actions.append("Suspend trading until compliance restored")
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error generating remediation actions: {e}")
            return []
    
    async def _log_audit_trail(self, 
                             user_id: str,
                             action: str,
                             entity_type: str,
                             entity_id: str,
                             old_values: Dict[str, Any],
                             new_values: Dict[str, Any],
                             reason: str,
                             ip_address: str=None,
                             session_id: str=None):
        """Log audit trail entry"""
        try:
            if not self.enable_audit_trail:
                return
            
            entry=AuditTrail(
                entry_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                timestamp=datetime.now(),
                user_id=user_id,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                old_values=old_values,
                new_values=new_values,
                reason=reason,
                ip_address=ip_address,
                session_id=session_id
            )
            
            self.audit_trail.append(entry)
            
            # Store in database
            with sqlite3.connect(self.compliance_db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_trail 
                    (entry_id, timestamp, user_id, action, entity_type, entity_id, 
                     old_values, new_values, reason, ip_address, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.user_id,
                    entry.action,
                    entry.entity_type,
                    entry.entity_id,
                    json.dumps(entry.old_values),
                    json.dumps(entry.new_values),
                    entry.reason,
                    entry.ip_address,
                    entry.session_id
                ))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error logging audit trail: {e}")
    
    async def generate_regulatory_report(self, 
                                       report_type: str,
                                       period_start: datetime,
                                       period_end: datetime,
                                       data: Dict[str, Any]) -> str:
        """
        Generate regulatory report
        
        Args:
            report_type: Type of report
            period_start: Report period start
            period_end: Report period end
            data: Report data
            
        Returns:
            str: Report ID
        """
        try:
            report_id=f"report_{report_type}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}"
            
            # Generate report data
            report_data={
                "report_id":report_id,
                "authority":self.primary_authority.value,
                "report_type":report_type,
                "period_start":period_start.isoformat(),
                "period_end":period_end.isoformat(),
                "generated_date":datetime.now().isoformat(),
                "data":serialize_datetime(data)  # Serialize any datetime objects in data
            }
            
            # Store in database
            with sqlite3.connect(self.compliance_db_path) as conn:
                conn.execute("""
                    INSERT INTO regulatory_reports 
                    (report_id, authority, report_type, period_start, period_end, 
                     status, data, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    report_id,
                    self.primary_authority.value,
                    report_type,
                    period_start.isoformat(),
                    period_end.isoformat(),
                    "generated",
                    json.dumps(report_data),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            # Log audit trail
            await self._log_audit_trail(
                user_id="system",
                action="generate_report",
                entity_type="report",
                entity_id=report_id,
                old_values={},
                new_values={"report_type":report_type, "period":f"{period_start} to {period_end}"},
                reason=f"Generated {report_type} regulatory report"
            )
            
            self.logger.info(f"Generated regulatory report: {report_id}")
            
            return report_id
            
        except Exception as e:
            self.logger.error(f"Error generating regulatory report: {e}")
            return ""
    
    async def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary"""
        try:
            # Get recent checks
            recent_checks=[c for c in self.compliance_checks 
                           if c.timestamp >= datetime.now() - timedelta(days=7)]
            
            # Count by status
            status_counts={}
            for status in ComplianceStatus:
                status_counts[status.value] = len([c for c in recent_checks if c.status == status])
            
            # Get critical violations
            critical_violations=[c for c in recent_checks 
                                 if c.status == ComplianceStatus.CRITICAL]
            
            summary = {
                "authority":self.primary_authority.value,
                "total_rules":len(self.compliance_rules),
                "active_rules":len([r for r in self.compliance_rules.values() if r.is_active]),
                "checks_run":self.check_count,
                "violations":self.violation_count,
                "last_check":self.last_compliance_check.isoformat() if self.last_compliance_check else None,
                "recent_checks":len(recent_checks),
                "status_counts":status_counts,
                "critical_violations":len(critical_violations),
                "audit_trail_entries":len(self.audit_trail)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting compliance summary: {e}")
            return {"error":str(e)}
    
    def check_position_compliance(self, positions: Dict[str, Any]) -> 'ComplianceCheck':
        """Check position compliance against rules
        
        Args:
            positions: Dictionary of positions to check
            
        Returns:
            ComplianceCheck: Result of compliance check
        """
        try:
            # Calculate portfolio metrics
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            max_position=max(pos.get('value', 0) for pos in positions.values()) if positions else 0
            concentration_risk=max_position / total_value if total_value > 0 else 0
            
            # Check against position limit rules
            violations = []
            for rule_id, rule in self.compliance_rules.items():
                if rule.rule_type== ComplianceRule.POSITION_LIMITS:
                    if concentration_risk > rule.threshold:
                        violations.append(f"Position concentration {concentration_risk:.2%} exceeds limit {rule.threshold:.2%}")
            
            # Determine compliance status
            status=ComplianceStatus.NON_COMPLIANT if violations else ComplianceStatus.COMPLIANT
            
            # Generate remediation actions if needed
            remediation_actions = []
            if violations:
                remediation_actions.append("Reduce largest position size")
                remediation_actions.append("Diversify portfolio across more assets")
                if concentration_risk > 0.3:
                    remediation_actions.append("URGENT: Position exceeds 30% - immediate action required")
            
            return ComplianceCheck(
                check_id=f"pos_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id="position_limits",
                timestamp=datetime.now(),
                status=status,
                current_value=concentration_risk,
                threshold_value=0.2,  # Default position limit
                deviation=max(0, concentration_risk - 0.2),
                severity="high" if violations else "low",
                details={"violations":violations, "concentration_risk":concentration_risk},
                remediation_actions=remediation_actions
            )
            
        except Exception as e:
            self.logger.error(f"Error in position compliance check: {e}")
            return ComplianceCheck(
                check_id=f"pos_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id="position_limits",
                timestamp=datetime.now(),
                status=ComplianceStatus.NON_COMPLIANT,
                current_value=0.0,
                threshold_value=0.2,
                deviation=0.0,
                severity="critical",
                details={"error":str(e)},
                remediation_actions=["Review system configuration", "Check data integrity"]
            )
    
    def add_compliance_rule(self, rule: ComplianceRuleDefinition):
        """Add new compliance rule"""
        try:
            self.compliance_rules[rule.rule_id] = rule
            
            # Log audit trail
            asyncio.create_task(self._log_audit_trail(
                user_id="admin",
                action="add_rule",
                entity_type="compliance_rule",
                entity_id=rule.rule_id,
                old_values={},
                new_values={"rule_id":rule.rule_id, "description":rule.description},
                reason="Added new compliance rule"
            ))
            
            self.logger.info(f"Added compliance rule: {rule.rule_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding compliance rule: {e}")
    
    def update_compliance_rule(self, rule_id: str, updates: Dict[str, Any]):
        """Update compliance rule"""
        try:
            if rule_id not in self.compliance_rules:
                raise ValueError(f"Rule {rule_id} not found")
            
            old_rule=self.compliance_rules[rule_id]
            
            # Update rule
            for key, value in updates.items():
                if hasattr(old_rule, key):
                    setattr(old_rule, key, value)
            
            old_rule.last_updated=datetime.now()
            
            # Log audit trail
            asyncio.create_task(self._log_audit_trail(
                user_id="admin",
                action="update_rule",
                entity_type="compliance_rule",
                entity_id=rule_id,
                old_values={"threshold":old_rule.threshold, "severity":old_rule.severity},
                new_values=updates,
                reason="Updated compliance rule"
            ))
            
            self.logger.info(f"Updated compliance rule: {rule_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating compliance rule: {e}")


# Example usage and testing
if __name__== "__main__":
    async def test_regulatory_compliance():
        # Initialize compliance manager
        compliance_manager=RegulatoryComplianceManager(
            primary_authority=RegulatoryAuthority.FCA,
            enable_audit_trail=True
        )
        
        # Simulate portfolio data
        portfolio_data={
            "positions":{
                "AAPL":{"value":15000, "quantity":100},
                "SPY":{"value":25000, "quantity":50},
                "TSLA":{"value":10000, "quantity":25}
            },
            "capital_ratio":0.12
        }
        
        # Simulate risk metrics
        risk_metrics={
            "portfolio_var":0.06,  # 6% VaR
            "portfolio_cvar":0.08,
            "concentration_risk":0.35
        }
        
        # Run compliance checks
        checks=await compliance_manager.run_compliance_checks(portfolio_data, risk_metrics)
        
        print(f"Ran {len(checks)} compliance checks")
        for check in checks:
            print(f"  {check.rule_id}: {check.status.value} "
                  f"(Current: {check.current_value:.2%}, Threshold: {check.threshold_value:.2%})")
        
        # Generate regulatory report
        report_id=await compliance_manager.generate_regulatory_report(
            "daily_risk_report",
            datetime.now() - timedelta(days=1),
            datetime.now(),
            {"risk_metrics":risk_metrics, "portfolio_data":portfolio_data}
        )
        
        print(f"Generated report: {report_id}")
        
        # Get compliance summary
        summary=await compliance_manager.get_compliance_summary()
        print(f"Compliance Summary: {summary}")
    
    asyncio.run(test_regulatory_compliance())


